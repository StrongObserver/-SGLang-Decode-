#!/usr/bin/env python3
"""Read-only syntax/manifest inspection. No imports of project code, no compilation."""
import ast
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
files = [p for folder in ("src", "benchmarks", "tests", "scripts") for p in (root/folder).rglob("*.py")]
for path in files:
    ast.parse(path.read_text(), filename=str(path))
for path in (root/"configs").glob("*.json"):
    json.loads(path.read_text())
subprocess.run(["git", "diff", "--check"], cwd=root, check=True)
print(f"Read-only AST parse: {len(files)} Python files; JSON and whitespace inspected.")
print("No project import, bytecode compilation, Triton JIT, tests, or GPU execution performed.")
