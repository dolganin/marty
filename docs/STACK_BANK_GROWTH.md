# Growing the frozen stack bank

`mars-rover-grow-stack-bank` is the only supported path for adding a mechanic
stack. It records a complete candidate trail under
`artifacts/stack_bank_growth/` and refuses to apply a candidate before all
five filters pass.

Install the two packages:

```bash
.venv/bin/pip install -e './environment' -e './baselines[train]'
```

Run one 50-candidate batch with pinned evaluator weights:

```bash
.venv/bin/mars-rover-grow-stack-bank \
  --batch 1 --count 50 \
  --robust-model /absolute/path/robust_ppo.zip \
  --recurrent-model /absolute/path/recurrent.pt \
  --apply --commit
```

The default `--proposal-source curated` uses the in-session model's reviewed
candidate catalogue and needs no API key. For an external proposal batch, use
`--proposal-source openai` and provide `OPENAI_API_KEY` only to that shell.
For an auditable replay, save an OpenAI response and use
`--input-json response.json`. The tool stops before requesting a new batch
when the 30/20/10 target or the 500-candidate exhaustion condition is already
met. `--apply` rewrites the frozen C++ stack table and manifest,
invalidates prior model provenance, and therefore requires rebuilding the
native extension and re-running the baseline qualification.

`--commit` stages only the current batch's files (forced because `artifacts/`
is ignored), plus the bank source and manifest when `--apply` changed them. It
does not stage unrelated worktree changes.
