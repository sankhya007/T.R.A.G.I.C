## What does this change?

A short description of what this PR does and why.

## Which model/script does this affect?

- [ ] SFM
- [ ] RVO
- [ ] Continuum Crowds
- [ ] Cellular Automata
- [ ] Parser / U-Net
- [ ] Zone editor
- [ ] Launcher GUI
- [ ] Docs only

## How was this tested?

What you actually ran to confirm it works (sample mask + config used, what the output looked like, etc). No formal test suite exists yet, so this is the important part.

## Checklist

- [ ] Ran the affected script(s) end-to-end and confirmed valid output
- [ ] If touching `Tragic_launcher.py`, clicked through the relevant GUI flow
- [ ] New parameters (if any) follow the existing `CONFIG`/`apply_runtime_args()` pattern
- [ ] No new files written outside `output/` (besides `stitched_mask.png` / zone config)
- [ ] Related issue linked below (if one exists)

Closes #