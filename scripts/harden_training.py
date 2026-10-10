"""Idempotent dataset and failure-diagnostic safety patches."""
from pathlib import Path
p=Path('src/experiments/training_runner.c')
s=p.read_text().replace('length<32','length<48').replace('length < 32','length < 48')
s=s.replace('at least 32 bytes','at least 48 bytes').replace('double loss;','double loss = 0.0;')
p.write_text(s)
p=Path('test/training_extra_test.c')
s=p.read_text().replace('CLOSE(dg[j],0)','CLOSE(dg[j],0.0)').replace('CLOSE(dg[j], 0)','CLOSE(dg[j], 0.0)')
s=s.replace('CLOSE(norm,3)','CLOSE(norm,3.0)').replace('CLOSE(norm, 3)','CLOSE(norm, 3.0)')
s=s.replace('CLOSE(d,1)','CLOSE(d,1.0)').replace('CLOSE(d, 1)','CLOSE(d, 1.0)')
p.write_text(s)
