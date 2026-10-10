"""Fix source-maintenance guards without invoking CUDA tools."""
from pathlib import Path
p=Path('scripts/finish_runtime_options.py');s=p.read_text()
s=s.replace("s=s.replace('tokens * sizeof(size_t) + sizeof(int)',", "if 'm->config.batch * sizeof(Segment)' not in s:\n s=s.replace('tokens * sizeof(size_t) + sizeof(int)',")
p.write_text(s)
