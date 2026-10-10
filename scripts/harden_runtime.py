## @file scripts/harden_runtime.py
# @brief harden runtime source/document maintenance tool.
# @ingroup maintenance
# @details Textual maintenance only; does not itself validate numerical correctness.

"""Idempotent runtime test hardening and documentation capability updates."""
from pathlib import Path
p=Path('test/runtime_test.c');s=p.read_text()
s=s.replace('CHECK(fputc(0xff,f)!=EOF);','int original=fgetc(f);CHECK(original!=EOF);CHECK(fseek(f,100,SEEK_SET)==0);CHECK(fputc(original^1,f)!=EOF);')
s=s.replace('CHECK(fputc(0xff, f) != EOF);','int original = fgetc(f); CHECK(original != EOF); CHECK(fseek(f, 100, SEEK_SET) == 0); CHECK(fputc(original ^ 1, f) != EOF);')
p.write_text(s)
for name in ('README.md','TRAINING.md','ARCHITECTURE.md'):
 p=Path(name);s=p.read_text()
 if '## Runtime lifecycle update' not in s:
  s+='''
## Runtime lifecycle update
The previous no-checkpoint, single-window, dropout-disabled and no-KV-cache
limitations are superseded by [RUNTIME.md](RUNTIME.md) and `models/runtime.h`.
Use `cfwx_run` for dropout-enabled mini-batch training, complete held-out evaluation,
checkpoint/resume and incremental cached generation. Earlier descriptions refer to
the original `cfwx_train` runner. CPU micro-batch accumulation is implemented;
fused batching and GPU execution remain explicitly unsupported rather than mislabeled.
'''
 p.write_text(s)
