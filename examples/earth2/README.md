# Earth-2 examples

All five short examples completed on the 16 GB proof host. Weather timings
use receipt-checked acquisition reuse. Native maps were inspected for all four
weather recipes. Export uses the weights-free `derived-ws10m` diagnostic;
Wind Gust AFNO refuses its recorded NVIDIA Community Model License weights.

Run from the repository root with a WOOF installation:

```sh
examples/earth2/forecast.sh --short --out /workspace/earth2-forecast
examples/earth2/storm_nest.sh --short --fetch-receipt /path/to/earth2-run.json --out /workspace/earth2-storm
examples/earth2/mixed_ensemble.sh --short --out /workspace/earth2-mix
python3 examples/earth2/hf_fetch.py --model fcn3 --out /workspace/earth2-weights
examples/earth2/export_diagnostic.sh --short --out /workspace/earth2-export
```

The forecast scripts use `woof earth2`; set `WOOF_BIN=gpuwm` for an engine
installation. `--short` requests 12 hours and selects UCast for the regional
and mixed examples. The short mix is `ucast:1,gefs:1`; `--model ID` overrides
the model and the AI source in the mix. UCast has a 12 hour cadence.
UCast inference fitted the 16 GB card at 4.746 GiB allocator peak. The 12 hour
regional run with receipt-checked forcing reuse completed in 146.3 seconds
at 4,467 MiB peak device memory above baseline and wrote nine maps.
The short two-member mix completed in 284.2 seconds at 6,633 MiB peak device
memory above baseline, also with receipt-checked acquisition reuse. Both
members completed 12 hours; wind and reflectivity sheets were reviewed at
0, 6 and 12 hours. These two members do not establish ensemble calibration.
SFNO inference
failed on this card, requesting 1.49 GiB with 1.36 GiB free.
The storm example keeps FCN3 and requests six hours with `--short`.
That version requires `--fetch-receipt FILE` from the completed larger-card fetch.
Without `--short`, the requests are 24 hours with FCN3 and the mix is
`fcn3:18,gefs:18`. Full-duration settings have not been measured here.

FCN3 inference needs a 32 GB card or larger (it allocates 20.4 GiB at once;
16 GB cards refuse). The storm short run needs a verified fetch from a larger card.

`hf_fetch.py` reads the weights repository from the authority and calls
`huggingface_hub.snapshot_download(token=False)`. It refuses an ID whose
row declares gated weights or an account requirement. It needs the
`huggingface_hub` package and a new output directory.

The six hour storm nest completed in 297.7 seconds at
3,218 MiB peak device memory above baseline, after verified FCN3 receipt reuse.
Its automatic center was 16.396 N, 99.821 W. The 12 hour export and DerivedWS
example, including the native diagnostic map, completed in
151.2 seconds at 2,926 MiB peak device memory above baseline.
These proofs used Earth2Studio 0.19.0 selected by `GPUWM_EARTH2_PYTHON`;
the storm tracker used its cyclone extra and CuPy in that isolated environment.
