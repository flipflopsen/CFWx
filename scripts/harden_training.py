"""Idempotent dataset length and failure-path diagnostic safety patches."""
from pathlib import Path
p=Path('src/experiments/training_runner.c')
s=p.read_text().replace('length<32','length<48').replace('length < 32','length < 48')
s=s.replace('at least 32 bytes','at least 48 bytes').replace('double loss;','double loss = 0.0;')
p.write_text(s)
