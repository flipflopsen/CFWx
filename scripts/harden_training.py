"""Apply small dataset/training safety fixes idempotently before validation."""
from pathlib import Path
p=Path('src/experiments/training_runner.c');s=p.read_text().replace('length<32','length<48').replace('length < 32','length < 48').replace('at least 32 bytes','at least 48 bytes').replace('double loss;','double loss = 0.0;');p.write_text(s)
p=Path('scripts/integrate_training.py');s=p.read_text();s=s.replace("from pathlib import Path", "from pathlib import Path\nimport runpy\nrunpy.run_path('scripts/harden_training.py')")
# Avoid appending the capability note on each maintenance run.
s=s.replace("s+='''\\n## Analytic training update", "\nif '## Analytic training update' not in s:\n s+='''\\n## Analytic training update")
p.write_text(s)
