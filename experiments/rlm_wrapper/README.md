## RLM Wrapper Experiment Track

This directory isolates experiments that use the external RLM wrapper (for example, from `alexzhang13/rlm`) so your current local RLM pipeline stays unchanged.

### Why keep this separate

- Fair comparison: "local RLM implementation" vs "official wrapper path"
- Cleaner paper narrative: avoids mixing implementation effects
- Lower risk: no regressions in your existing scripts

### Suggested protocol

1. Keep your existing benchmarks unchanged.
2. Run a smoke test with `run_wrapper_probe.py`.
3. Implement a wrapper-backed runner that matches one existing task at a time:
   - Oolong numeric aggregation
   - 19-tab extraction benchmark
   - synthetic structured extraction
4. Record outputs in `results/` using a distinct filename prefix (for example, `wrapper_*`).

### Quick start

```powershell
python "c:\Users\Rhucha Sant\rlm-vs-rag-research\experiments\rlm_wrapper\run_wrapper_probe.py" --query "What is 2+2?" --context "Simple math context"
```

### Notes

- The probe script intentionally does not assume one fixed wrapper API.
- If the wrapper package is not installed, it prints next steps and exits cleanly.
- After you confirm import + basic invocation, we can wire it into your benchmark scripts.
