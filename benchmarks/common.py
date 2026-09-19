"""Raw sample/manifest recording. No historical numbers are emitted as results."""
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import torch
import triton


def save_run(directory, rows, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        sha = "UNKNOWN"
    manifest = dict(metadata, personal_patch_sha=sha, python=platform.python_version(),
                    torch=torch.__version__, triton=triton.__version__,
                    device=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                    quantile_method="median of round means; range is not P99",
                    upstream_sha="7e257cd666c0d639626487987ea8e590da1e9395")
    manifest["workload_hash"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
