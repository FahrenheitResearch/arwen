# Carried physics change channel

`gpuwm-carried-channel` is an offline command for exchanging scoped engine
release bytes and reviewing differences against a consumer's carried source.
It calls `tools.carried_physics_channel:main`. The identical command can be run
from a source tree as `python tools/carried_physics_channel.py`.

The tool never merges physics, rewrites an existing report, publishes an asset,
loads executable code from an artifact, or calls a GPU. A receiver is trusted
local Python code explicitly selected by its filesystem path. Serialize receiver
loading if calling the API from a multithreaded application.

## Identity and coverage

The consumer owns a typed mapping of engine file and directory paths to carried
paths. Directory units are not expanded from the carried side alone. The Git
producer enumerates both immutable endpoint trees, unions their scoped members,
and captures raw bytes for every named member. Unchanged files are included;
`null` means proven absence, not an omitted inventory row. Direct API callers
must supply equally scoped complete snapshots, with explicit absence for every
missing file. Partial maps cannot establish that a directory is complete.

The producer reads regular Git blobs, not worktree bytes, checkout filters or
symlinks. The current consumer does not define transitive import closure. A new
helper outside its mapping needs a mapping review before inclusion; a new file
inside a mapped directory is covered without editing a file allowlist.

The release schema is `arwen.carried-release.v2`. Its scope is
`arwen.carried-scope.v1`; each unit has `engine`, `carried` and `kind` (`file`
or `tree`). Old and new identities carry complete commit/tree object ids,
commit timestamps and explicit version labels. `verify-git` checks that the
named objects reproduce every scoped blob, absent member and directory child.
It does not prove that a version label matches package metadata or that the
commits are approved publication endpoints. The release preparation owner must
check those separately.

Release hunk descriptors are diagnostic old/new raw byte ranges. Their hashes
are never consumer keys. Actual consumer keys are measured from a complete
engine/carried pair using the consumer's own rules. Text coordinates belong to
the whole-file rewritten, LF-normalised engine sequence. The byte API accepts
only named `engine_raw` and `carried_raw` arguments. Empty files, absent files,
terminal split elements, binary fallback and default diff matching retain the
receiver's v1 semantics. Rewiring still precedes newline conversion.

## Trust inputs

A content id proves integrity relative to an expected id, not authorship by
itself. Every review, candidate export and feedback verification requires
`--expected-release-id`. Obtain that value from a local emission you performed,
or from an independently authenticated publication manifest. Reading the id
out of the same received JSON is not authentication.

Git verification and independent id verification serve different purposes:
the first checks scoped bytes and coverage against named Git objects; the
second pins which release object the caller intended to review. Neither proves
numerical correctness, an evidence reference's contents, or a reviewer's intent.

`classify` additionally requires the selected review id. A classified receipt
keeps the original review, all explicit annotations and the retired-row
acknowledgements. Candidate export and feedback generation recompute the review
against the release and the complete current carried snapshot before using it.
A stale baseline cannot pass because its version label happens to match:
previous engine paths and the complete multiset of fingerprint keys must match
the actual old-engine/current-carried measurement under the recorded rules.

The legacy registry has no historical normalization provenance. Matching its
keys is reported as a measurement under the recorded current rules, not proof
of which earlier source produced it. Receiver source, rule source, diff library
and Python identity are recorded for each new review. A change to those pins
requires remeasurement, not a forecast admission refusal.

## Commands

Use absolute paths for `ENGINE` and `GLOBAL`. Set `OLD_REF`, `NEW_REF`,
`OLD_VERSION` and `NEW_VERSION` to the intended immutable endpoints and labels.
`OUT` must name an existing output directory. Explicit output filenames must
not already exist. Omitting `--out` allocates a new generation under
`carried-channel-output/` and prints its path and id, so retries do not overwrite
previous evidence.

```sh
TOOL="$ENGINE/tools/carried_physics_channel.py"
RECEIVER="$GLOBAL/tools/fingerprint_engine_divergence.py"
CARRIED="$GLOBAL/src/arwen_global"

python "$TOOL" scope --receiver "$RECEIVER" --out "$OUT/scope.json"
python "$TOOL" emit --repo "$ENGINE" --scope "$OUT/scope.json" \
  --old "$OLD_REF" --new "$NEW_REF" \
  --old-version "$OLD_VERSION" --new-version "$NEW_VERSION" \
  --out "$OUT/release.json"
python "$TOOL" verify-git --repo "$ENGINE" --manifest "$OUT/release.json"
```

When the published baseline is in a separate mirror, pass `--old-repo` to
`emit` and `verify-git`. Both endpoints are read from Git objects; neither
checkout is changed or fetched into the other.

### Release packet attachment

The cut uses `tools/release/prepare_carried_release.py`, which obtains version
labels from each endpoint's `pyproject.toml` and independently verifies all
scoped bytes. Run it against the final release source revision:

```sh
python "$ENGINE/tools/release/prepare_carried_release.py" emit \
  --repo "$ENGINE" --old-repo "$PUBLIC_BASELINE" \
  --old "$OLD_REF" --new "$NEW_REF" --receiver "$RECEIVER" --out "$NEW_OUTPUT"
```

This produces `gpuwm-carried-physics-vOLD-vNEW.json` and its matching
`.verification.json` sidecar. The local packet finalizer takes them through
`--carried-channel` and `--carried-channel-verification`, verifies them against
the release source, and hashes both into `PUBLICATION-ASSETS.json` under
`carried_physics`. The verification archive includes the portable sidecar.
Starting with 2.7.4, promotion refuses a missing channel, missing sidecar,
changed packet identity or mismatched release revision. Its proof rechecks the
actual Git endpoint bytes. This prevents a corrective release from silently
omitting changes to its consumers' carried files; it does not approve those
changes for a consumer or alter any forecast admission rule.

Set `RELEASE_ID` to the id printed by that trusted local emission, or to the
independent publication pin when reviewing a received release. Then measure:

```sh
python "$TOOL" review --manifest "$OUT/release.json" \
  --expected-release-id "$RELEASE_ID" --receiver "$RECEIVER" \
  --carried "$CARRIED" --previous "$CARRIED/data/engine-divergence.json" \
  --out "$OUT/review.json"
```

Review returns status 1 when new, retired, conflicting or unclassified rows need
attention, after writing the report. Status 2 is invalid input or an operation
failure. Do not silently replace a rejected baseline with a different one.
Investigate endpoint, carried-source and rule changes. An explicit `--bootstrap`
in place of `--previous` starts with no inherited classifications and records
that fact. Preserve the superseded registry separately and explain any reset;
bootstrap is not a migration of old decisions.

Read the report and create a decisions JSON. `review_id` must be the exact
review id. This illustrates one new row and one retired row, not a blanket
classification instruction:

```json
{
  "schema": "arwen.carried-decisions.v2",
  "review_id": "<selected review id>",
  "acknowledged_retired": [0],
  "rows": [{
    "index": 0,
    "row": "SCHEME-1",
    "class": "global-fix",
    "decision": "offer",
    "note": "Describe the observed behavior and the reason for this decision."
  }]
}
```

Every retired row needs exactly one acknowledgement. Conflicting historical
annotations on an identical key require an explicit decision at every current
occurrence. Multiplicity is preserved; rows are not collapsed into a dictionary
entry. New rows remain unknown until authored annotations cover them. Only
`row`, `class`, `decision` and `note` can be edited. This channel does not infer
scientific classes from byte equality or silence.

```sh
python "$TOOL" classify --review "$OUT/review.json" \
  --expected-review-id "$REVIEW_ID" --decisions "$OUT/decisions.json" \
  --receiver "$RECEIVER" --out "$OUT/classified.json"
python "$TOOL" candidate --manifest "$OUT/release.json" \
  --expected-release-id "$RELEASE_ID" --receiver "$RECEIVER" \
  --carried "$CARRIED" --classified "$OUT/classified.json" \
  --out "$OUT/engine-divergence.candidate.json"
```

The candidate uses the receiver's real legacy generator. It does not overwrite
the installed registry or its document, and does not itself prove that the
document has been updated. Retain the classified receipt with it. Existing
registry/document gates must pass before adopting a candidate. The old receiver
`--rewrite` remains available with its established overwrite and status behavior;
it is not a no-mutation channel command.

## Feedback

Feedback is generated from a fully classified receipt. Select only the exact
row occurrences to offer, with behavioral evidence references:

```sh
python "$TOOL" feedback --manifest "$OUT/release.json" \
  --expected-release-id "$RELEASE_ID" --receiver "$RECEIVER" \
  --carried "$CARRIED" --classified "$OUT/classified.json" \
  --index 0 --evidence "<behavioral evidence reference>" \
  --out "$OUT/feedback.json"
```

This exports metadata, annotations and raw-file descriptors, not carried source
bytes. An annotation or evidence string can itself contain sensitive text, so
review it before sharing. To intentionally include a whole selected file, add
`--include-source core/example.py`, using a file that has a selected offer.
Repeat for other selected files. No unrelated source file can be included.
There is no automatic reverse rewrite, engine patch generation or partial apply.
An engine-native correction must still be authored and reviewed separately.

```sh
python "$TOOL" verify-feedback --manifest "$OUT/release.json" \
  --expected-release-id "$RELEASE_ID" --receiver "$RECEIVER" \
  --feedback "$OUT/feedback.json"
```

The result distinguishes `not-recomputed`, `partially-recomputed` and
`recomputed` pair identities. Full pair verification requires complete bytes for
every selected file, either explicitly included in feedback or independently
supplied with `--carried`. Add `--require-pairs` to fail when any are missing.
The verifier checks the receiver's actual path mapping, whole-pair measurement,
transformed coordinates and per-file occurrence ordinal, not only side hashes.

`--classified` checks the claimed global review indexes and annotations against
a supplied classification receipt. Without it, `review_membership` is
`not-checked`, even when pair membership is recomputed. `--expected-feedback-id`
pins the feedback to an independently trusted id. Without that pin,
`feedback_identity` is `not-authenticated`. An internally valid receipt is not
proof of its author's approval; behavioral evidence always requires review.

## CPU validation

```sh
# Engine checkout
python -m unittest discover -s tests -p test_carried_channel.py -v

# Global checkout
ARWEN_CARRIED_CHANNEL="$ENGINE/tools/carried_physics_channel.py" \
  python -m unittest discover -s tests -p test_carried_channel.py -v
```

Both commands avoid importing either physics runtime. The integration suite
uses the actual receiver against synthetic source snapshots. It is not a full
repository or numerical qualification run. Historical oracle vectors must be
checked under their historical rules; current-rule differentials use the
untouched current receiver as their comparison, never rewritten old expectations.

## Publication boundary

The engine console entry is declared in `pyproject.toml`, and the existing
publication test job includes the CPU transport suite. This is not automatic
release-asset attachment. The publication workflow promotes a prepared packet
without rebuilding its bytes. Generate and verify the scoped release object
before that packet is sealed, then bind its filename, digest, endpoint ids and
scope to the preparation owner's asset manifest. It is a supplementary release
asset, not another PyPI distribution.

`tools/promote_prepared_release.py` owns the workflow's `capture`, `verify`,
`promote` and final-publication checks. Its implementation and the preparation
guide `docs/dev/prepared-publication.md` are needed to locate the exact preparer
and extend its asset inventory. Do not generate or attach new bytes inside the
OIDC-only PyPI upload job. Full wheel/sdist and public-asset qualification remain
separate from this offline tool's tests.
