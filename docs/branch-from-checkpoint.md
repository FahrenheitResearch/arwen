# What if? Branching a run from a checkpoint

`gpuwm resume` continues the run that wrote the checkpoint: same config,
same output directory, one trajectory. `gpuwm branch` answers the other
question — *what would this forecast have done from hour 3 if the
tracker were wider, or if the history tape were trimmed?* — by starting
a **new** run from the first run's saved state and writing it somewhere
else entirely.

The source run is opened read-only. Nothing is written into it, ever.

```
gpuwm branch myrun.toml --from-run out/myrun --outdir out/myrun-wider --set relocation.follow.radius_km=80
```

That produces `out/myrun-wider/` holding

| file | what it is |
| --- | --- |
| `branch-config.toml` | the branched configuration — the new run's authority, the same bytes that integrate |
| `branch_receipt.json` | the decision record: parent checkpoint, every setting that moved and what it moved from, and the identity comparison |
| `branch_manifest.json` | the bytes record: every member of the parent checkpoint set, sized and hashed |

and then it integrates, exactly as `gpuwm run` would.

## What a branch may change, and why the rest is refused

A checkpoint is a state that was computed under a configuration. Most of
that configuration is therefore part of what the state **is**: restore it
under different physics, a different grid, or different forcing and the
numbers in the file no longer mean what the new run would assume they
mean. The restart machinery already refuses that, at restore time — which
is to say after the fetch, the static build and the device allocation.
`gpuwm branch` refuses it at the door instead, by name, with the reason.

Changeable from a checkpoint:

| setting | why it is free |
| --- | --- |
| `run_seconds`, `restart_interval_s` | how far the run goes and how often it saves — the published `--restart` tolerance |
| `acknowledgements` | a governance declaration, not a computed value |
| `relocation.*` | tracker **bounds**: which child may move, how far, how often, and the `[relocation.follow]` / `[relocation.containment]` / `[relocation.track]` blocks |
| `tiles.*` | resident vs streamed execution, which by contract changes no bytes |
| `output.*` | which variables reach the history tape; checkpoints are written from model state, never from the history frame |
| `domain.<grid_id>.history_interval_s` | per-domain write cadence |
| `domain.<grid_id>.tiles.*`, `domain.<grid_id>.output.*` | the per-domain halves of the two above |

Everything else is refused with the breakage named. The one split worth
learning, because a what-if screen will reach for the wrong half:

- `[relocation.follow]` is the tracker's **bounds** — how it hunts. Free.
- `[follow]` on a `[[domain]]` is where that child **sits**, and
  `[retire]` / `[rearm]` / `[spawn]` decide whether it integrates at all.
  Those bind: a branch that changed one would restore state computed
  under a nest history that did not happen.

A refusal names the setting, the breakage, the changeable list and the
way out (start the experiment from t=0 for a change the checkpoint
cannot honour). Nothing is created when a branch is refused — the target
directory is exactly as it was, so the corrected retry works.

## Choosing the checkpoint

`--from-run` names the source run's output directory and `--from`
chooses within it: `latest` (the default) takes the newest set whose
members all validate, and skipped newer sets are printed with the reason
they were skipped. Pass an explicit `gpuwmrst_*.npz` path to `--from`
to branch from a particular instant:

```
gpuwm branch myrun.toml --from out/myrun/gpuwmrst_d01_2011-04-27_21_00_00.npz --outdir out/myrun-21z
```

Checkpoint discovery is the same code `gpuwm resume` uses, so the two
doors cannot disagree about which file is the newest valid one.

## Looking before you leap

`--prepare-only` writes the run directory, the branched config and both
receipts, then stops without integrating:

```
gpuwm branch myrun.toml --from-run out/myrun --outdir out/myrun-wider --set output.preset=minimal --prepare-only
```

Every refusal a branch can raise has already been raised at that point,
and `branch_receipt.json` says exactly what would run — which is what a
what-if dialog shows before it commits a card to the answer.

## What "the same, only different" means here

With no `--set` at all, a branch is a resume that writes elsewhere: the
branched config carries a byte-identical restart identity payload
(`gpuwm.core.model.restart_identity_payload`, the definition the
experiment fingerprint hashes) and the same checkpoint, so the same state
integrates under the same rules. The receipt records both payload
digests side by side, and they are equal.

Add a `--set` and only the named setting moves — the receipt's
`overrides` rows carry the before and after values, so the difference
between two runs is a thing you read rather than a thing you reconstruct.

## Declared inputs move with the config

A branched config lives in the new run directory, and `[case_data]`,
`[static]` and `[ingest]` paths are resolved against the config file's
own directory. Relative declarations are therefore rewritten to absolute
paths against the **source** config's directory on the way out, and each
rewrite is listed in the receipt's `rebased_paths`. A branch of a working
config points at the same bytes the parent ran on.
