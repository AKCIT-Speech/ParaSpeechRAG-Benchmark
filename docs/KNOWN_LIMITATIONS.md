# Known limitations and methodological conventions

This file records the conventions a replicator needs in order to use this code
correctly, and the places where a number produced here is not directly
comparable to one produced elsewhere. None of it is optional reading if you
intend to compare your results against ours.

For what the repository can and cannot evaluate, see the claim → command table
in the [README](../README.md). For dataset licensing, see
[`DATA_LICENSES.md`](DATA_LICENSES.md). For metric naming, see
[`NAMING.md`](NAMING.md).

---

## 1. The released audio is the artifact, not the generator code

No perturbation axis in this benchmark is bit-reproducible:

- **Voice conversion** — Seed-VC runs a 30-step diffusion sampler and GenVC
  samples with `top_k=15`. Neither is deterministic, and neither driver sets or
  logs a seed. A different torch or CUDA build yields different waveforms, and
  therefore different Recall@1.
- **Codec** — output depends on the local ffmpeg / libmp3lame / libopus build.
- **Rate** — output depends on the librosa phase-vocoder version.
- **Noise** — generation is seeded (`--noise-seed`, `--ambient-seed`), but still
  depends on the ESC-50 release and the local numpy version.

The reproducible path is therefore:

```
released audio  →  embeddings  →  metrics
```

which is deterministic. Generator code in `scripts/perturb/` and
`src/paraspeechrag/perturb/` is **provenance**, not the reproduction path.
`scripts/build_manifest.py` pins the first arrow with a SHA256 per file; verify
with `make verify-manifests` before trusting any comparison.

## 2. SNR is defined over full-signal RMS, not ITU-T P.56

`add_white_noise` and `add_ambient_noise`
(`src/paraspeechrag/perturb/noise.py`) compute:

```python
P_signal = mean(x ** 2)                     # includes silence and pauses
P_noise  = P_signal / 10 ** (snr_db / 10)
```

This is **not** the ITU-T P.56 active-speech level. Spoken-SQuAD paragraph
readings contain substantial silence, so full-signal RMS understates the speech
level and the effective speech-to-noise ratio is **higher** than the nominal
`snr_db`. Results will not line up against work using the P.56 convention.

Both additive functions also clip the sum to `[-1, 1]`, which at low SNR is a
nonlinear distortion layered on top of the additive noise.

Do not change this convention without regenerating every noise result.

## 3. Reverberation is parameterised by decay time, not SNR

`add_reverberation` takes a decay time. The evaluation driver
(`scripts/run_noise_robustness_eval.py:158`) reuses the `--snr-levels` values as
a strength knob via `decay_time_ms = snr * 10`. A row tagged `reverb_20.0` is a
**200 ms decay tail**, not a 20 dB SNR.

Three consequences for anyone reading or extending the reverb rows:

1. **Reverb rows do not share an axis with the white/ambient rows.** Plotting
   them against a common "noise level" axis is not meaningful, and the direction
   of severity is opposite: as the label falls 20 → 5 the tail *shortens*, so
   the perturbation gets *milder*.
2. **The swept range is narrow in effect.** RMS deviation from clean moves only
   ~5% across the full sweep (0.2965 → 0.2824), versus 5.6× for white noise over
   the same labels. The curve is flat because the knob is inert in this range.
3. **The endpoint differs structurally.** The fixed 50 ms early reflection is
   placed at an absolute offset, so for `decay_time_ms < 50` it falls outside
   the impulse response and is silently dropped. The mildest condition uses a
   different IR shape from the other three.

The IR is hand-written (exponential envelope, direct sound forced to 1.0, one
+0.5 early reflection), not measured or simulated, and the output is
peak-normalised to 0.95 — a gain change the additive functions never apply.
Convolution uses `mode="same"`, which centres the result and therefore
time-advances it by roughly half the IR length.

A reverberation study that is comparable to the literature needs a decay-time
axis or measured RIRs, and a re-run. The behaviour is preserved as-is here
because changing it would silently invalidate existing results.

## 4. Candidate pool size must be reported alongside any Recall@k

`collect_paragraph_chunks` **silently drops paragraphs with no audio**. A
partially generated perturbation directory therefore yields a small candidate
pool with no error raised, and Recall@1 over 91 candidates is not comparable to
Recall@1 over 2,067.

Guards in place:

- `scripts/sanity/audit_perturbation_coverage.py` (`make check-coverage`)
  reports paragraph coverage per perturbation directory.
- The PKL builder prints the pool size and warns on shortfall.
- `run_noise_robustness_eval.py` writes `n_candidates_in_pool` and `n_queries`
  into a `.meta.json` sidecar beside every results CSV.

Run `make check-coverage` before comparing any perturbed condition against the
clean baseline. A ΔR@1 computed across two different pool sizes mostly measures
the pool-size difference.

Note also that the GenVC driver's `should_include()` keeps only article 0 (all
paragraphs), article 10 (all) and article 11 paragraphs 0–13 — roughly 3k source
audios, **not** full dev-split coverage. The GenVC conditions therefore have a
smaller pool than the clean baseline by construction.

## 5. Ranking metrics assume one relevant item per query

`compute_ranking_metrics` (`src/paraspeechrag/eval/ranking_metrics.py`) assumes
a square similarity matrix with diagonal ground truth: query *i* is relevant to
candidate *i*. Two things follow.

- **The query count equals the candidate pool size** by construction. There is
  no separate query-set size to report.
- `Hits@k` ≡ `Recall@k` and `MAP` ≡ `MRR` under this assumption. See
  [`NAMING.md`](NAMING.md).

Ties are broken optimistically: `rank = 1 + |{j : score_j > score_gold}|`. With
many tied scores this reports the best-case rank.

## 6. Sampling rates differ in history, not at the model input

Different axes write different sample rates:

| Axis | Written at | Note |
|---|---|---|
| Seed-VC (speaker, emotion) | 22,050 Hz | `batch_convert.py:132`; never resampled on write |
| Opus codec | 24,000 Hz | forced `-ar 24000` on decode |
| MP3 codec | source rate | no `-ar` flag on decode |
| Rate / time-stretch | source rate | preserved by librosa |
| GenVC | its config's `audio.sample_rate` | — |
| Noise | 16,000 Hz | generated in-memory at the retriever's rate |

All of these are resampled to 16 kHz mono by
`paraspeechrag.inference.audio_preprocess` before reaching the retriever, so
they differ in **resampling history**, not in the rate the model sees. Retrieval
results are unaffected; audio-quality comparisons across axes are not
apples-to-apples.

## 7. Training runs are not reproducible

`scripts/train.py` sets no global `torch` / `numpy` / `random` seed, so training
is not reproducible even on identical hardware. Identify a model by its
checkpoint SHA256 rather than by its training command; record it in
`configs/retrievers/*.yaml`.

Provenance for the released checkpoint — epoch count, contrastive temperature,
and the exact Spoken-SQuAD training subset — is recorded in
`configs/train/clasp.yaml`, where TODO markers indicate values that were not
logged at training time.

## 8. Train / validation separation

Paragraph keys `{article}_{paragraph}` are positions within a *single* SQuAD
JSON, so `0_0` in `spoken_train-v1.1.json` and `0_0` in `spoken_test-v1.1.json`
are **different paragraphs with different context text**. Passing one perturbed
directory as both `--train-wav-dir` and `--val-wav-dir` therefore pairs the same
WAV files with different text across splits.

Guards in place:

- `scripts/build_spoken_squad_pkl.py` refuses identical train/val WAV
  directories unless `--allow-shared-wav-dir` is passed explicitly.
- `paragraph_id` is namespaced as `"{split}:{article}_{paragraph}"`.
- `scripts/sanity/check_split_disjoint.py` (`make check-splits`) asserts
  disjointness over a built PKL.

**Any PKL built before these guards existed should be rebuilt**, or at minimum
run through `make check-splits`.

## 9. Scope

The evaluation covers the **retrieval stage only**, consistent with the paper's
stated scope. Downstream generation quality under perturbation is not measured,
so how retrieval instability propagates to end-to-end answers is an open
question.

The benchmark is English-only (Spoken-SQuAD, VoxPopuli English). Perturbations
are synthetic — TTS-generated source audio, synthetic noise, and voice
conversion — and may not reflect naturally occurring acoustic degradation, which
limits ecological validity.
