# Seed-VC generators (speaker + emotion axes)

Vendored from the `SCRIPT_TEST/` folder of a Seed-VC checkout
(`seed-vc-test/`, ~1.1 GB) that lives outside this repo. Only the
project-specific scripts are here — **not** the upstream Seed-VC code, its
checkpoints, or the reference audio.

## Running these

They import from the Seed-VC repo root, so they must execute from inside a
Seed-VC checkout, not from here:

```bash
cd /path/to/seed-vc            # upstream: https://github.com/Plachtaa/seed-vc
cp -r /path/to/ParaSpeechRAG/scripts/perturb/seedvc SCRIPT_TEST
python SCRIPT_TEST/batch_convert.py --help
```

`batch_convert_v2.py` additionally reads `configs/v2/vc_wrapper.yaml` relative
to the Seed-VC root.

| File | Axis | What it does |
|---|---|---|
| `batch_convert.py` | speaker | Seed-VC v1 conversion to each target voice |
| `batch_convert_v2.py` | emotion | Seed-VC v2 with `convert_style=True` |
| `select_ravdess_references.py` | emotion | Picks RAVDESS reference clips per emotion/intensity |
| `run_ravdess_actor01_background.sh` | emotion | The exact invocation used for the paper's emotion set |
| `run_ravdess_actor01_invalids_background.sh` | emotion | Re-run helper for failed conversions |

## Reproducibility

Seed-VC runs a 30-step diffusion sampler and neither script sets or logs a
seed, so conversions are not bit-reproducible. The released **audio** is the
artifact; pin it with `scripts/build_manifest.py`.
