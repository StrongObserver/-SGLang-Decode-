"""Standalone CUDA Graph owner: stable pointers, replayed plan, layer-private scratch."""
import torch
from .engine import DecodeEngine


class DecodeGraph:
    def __init__(self, engine: DecodeEngine, keys, kv_pairs):
        if len(keys) != len(engine.layers) or len(kv_pairs) != len(keys):
            raise ValueError("one key and KV pair per layer required")
        if not keys or any(key != keys[0] for key in keys):
            raise ValueError("incompatible layers require separate plan/graph instances")
        self.engine, self.kv_pairs = engine, kv_pairs
        c, p = engine.config, engine.plan
        self.q = [torch.empty_like(layer.output) for layer in engine.layers]
        self.pages = [torch.zeros((c.batch_capacity, c.max_pages), dtype=torch.int32,
                                 device=p.device) for _ in keys]
        self.ready, self.done = torch.cuda.Event(), torch.cuda.Event()
        self.graph = None
        self.has_run = False

    def update(self, queries, page_tables, lengths, splits):
        """Host lengths are already known by the scheduler; no GPU scalar readback."""
        c, p = self.engine.config, self.engine.plan
        if not 0 < len(lengths) <= c.batch_capacity or len(splits) != len(lengths):
            raise ValueError("choose a larger graph or upstream before submitting")
        if any(x < 0 or x > c.max_context for x in lengths):
            raise ValueError("context outside graph capacity")
        if any(x < 1 or x > c.max_splits for x in splits):
            raise ValueError("split capacity exceeded")
        if len(queries) != len(self.q) or len(page_tables) != len(self.pages):
            raise ValueError("layer count mismatch")
        stream = torch.cuda.current_stream(p.device)
        if self.has_run:
            stream.wait_event(self.done)
        n = len(lengths)
        p.lengths.zero_()
        p.splits.fill_(1)
        p.lengths[:n].copy_(torch.tensor(lengths, device=p.device, dtype=torch.int32))
        p.splits[:n].copy_(torch.tensor(splits, device=p.device, dtype=torch.int32))
        p.active.fill_(n)
        for target_q, target_pages, q, pages in zip(self.q, self.pages, queries, page_tables):
            if q.shape != (n, c.query_heads, c.head_dim) or pages.shape != (n, c.max_pages):
                raise ValueError("input shape mismatch")
            target_q[:n].copy_(q)
            target_pages[:n].copy_(pages)
        self.ready.record(stream)

    def _step(self):
        self.engine.prepare()
        for i, (k, v) in enumerate(self.kv_pairs):
            self.engine.execute(self.q[i], k, v, self.pages[i], layer=i)

    def capture(self, warmup=3):
        """Explicit opt-in GPU operation. Never invoked at import or installation."""
        if self.graph is not None:
            raise RuntimeError("capture once; create another instance to change geometry")
        stream = torch.cuda.Stream(device=self.engine.plan.device)
        stream.wait_event(self.ready)
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._step()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self._step()
        self.done.record(stream)
        self.has_run = True

    def replay(self):
        if self.graph is None:
            raise RuntimeError("capture after initializing representative inputs")
        stream = torch.cuda.current_stream(self.engine.plan.device)
        stream.wait_event(self.ready)
        if self.has_run:
            stream.wait_event(self.done)
        self.graph.replay()
        self.done.record(stream)
        self.has_run = True
        return [layer.output for layer in self.engine.layers]
