# Robust + causal value-channel quantization

## Context

`open_optformer/history.py`'s `History.get_prompt()` (lines 131–147) tokenizes each trial's objective value by computing `y_min`/`y_max` as the raw `min()`/`max()` over `self.trials`, then linearly mapping into `[0, num_numeric_tokens-1]` via `quantize()` (lines 11–23). This single code path is used both to build training strings (`generate_training_data/load_data.py` → `History.from_syne_tune_experiment`, which loads the **entire** trajectory into `hist.trials` before `get_prompt()` runs once) and at inference (`optformer_searcher.py`, where `self.study` starts empty and trials are added one at a time via `on_trial_complete`, so `get_prompt()` is called repeatedly with a strictly growing prefix).

This produces two related but distinct problems:

1. **Outlier sensitivity (fcnet).** Raw min/max means one diverged fcnet run stretches the scale for the whole trajectory, wasting resolution on all other (well-behaved) trials.
2. **Train/inference scale mismatch (causality).** At train time, `y_min`/`y_max` are computed over the *full* trajectory (including trials that occur later in the sequence than the token being emitted) — non-causal. At inference, `y_min`/`y_max` are necessarily computed over only the trials observed so far — causal. So the model is trained on a value-token distribution it will never see at inference time (early-trajectory tokens are quantized against future information during training, but not during search).

No existing repo document has a "§2" describing this — the closest related note is the v0.5 quantization-bug entry in `README.md:156` (an off-by-one issue, not this one). A percentile-clip pattern already exists (commented out) in `benchmarks/syne_tune_benchmarks/results_analysis/show_results.py:384` and is the natural template to reuse for issue (1).

## Scientific evaluation

**Issue 2 (causality) is the higher-priority fix.** It's a structural train/inference distribution shift affecting *every* benchmark, not just fcnet: the model learns "token 500 means roughly this value" under a full-trajectory scale, but at inference (especially early in a run, with few observed trials) that scale is unstable and systematically different — often narrower, since it hasn't seen the trajectory's eventual extremes. This biases the model's read of early value tokens during search, which is exactly when the model has the least context and most needs a reliable value signal to drive exploration/exploitation. Fixing it (quantize against min/max of trials observed *up to and including* the current one during data generation) makes training and inference distributions match exactly, with no known downside — it's a strict correctness fix, not a tradeoff.

**Issue 1 (outlier clipping) is a smaller, second-order fix, and should be done asymmetrically rather than as a symmetric 5th–95th clip.** In `load_data.py:177-178`, the metric is already negated when `mode == 'max'`, so `History` always internally treats **lower metric = better**. That means the two tails carry very different information:

- The **upper (bad) tail** is where divergence/instability lives (e.g. a diverged fcnet run). This is exactly the outlier problem raised — a diverged run doesn't carry useful graded information beyond "avoid this region," so clipping it costs little and reclaims resolution for the rest of the trajectory.
- The **lower (good) tail** is where search decisions are most sensitive to precision — distinguishing the best few configs from the merely-good ones is usually the entire point of a BO run, especially near convergence. Clipping this side (as a symmetric 5th–95th scheme would) throws away exactly the resolution that matters most.

So a **symmetric** percentile clip is a worse trade than necessary: it fixes the outlier problem but also degrades fidelity among the best trials, which have no divergence problem to begin with. The better design clips only the tail opposite the optimization direction — cap `y_max` at a high percentile (e.g. 95th, tunable), while leaving `y_min` as the true observed min (or a much more conservative/high percentile, mainly as a rarely-triggered safety net rather than routine clipping). Since the max→min direction normalization already happens upstream of `History`, this asymmetric rule ("always clip the max side, leave the min side alone") is direction-agnostic and needs no per-benchmark metadata.

Expected effect: more precise ranking/token separation for the best-performing trials (the ones that matter most for search quality) and for the well-behaved majority, at the cost of coarser distinctions only among the small tail of diverged/catastrophic trials — which matters little since a searcher rarely needs to fine-rank already-bad configs.

**Combined effect on model performance:** Both changes should reduce value-token noise/variance without changing token vocabulary size or architecture. The causal fix should give a clean, low-risk improvement in calibration during search (especially early-trajectory decisions). The outlier fix should reduce fcnet-specific resolution collapse, likely improving loss and downstream regret specifically on fcnet-family benchmarks, with negligible effect elsewhere (since it's a no-op when no trial falls outside the percentile band). Recommend implementing and evaluating both, but attribute their effects separately (see verification) since they address independent failure modes and should be ablated independently.

## Implementation approach

Both changes are localized to `open_optformer/history.py`, primarily inside `get_prompt()`.

### 1. Causal min/max (train/inference consistency)

In `get_prompt()` (`history.py:131-148`), replace the single trajectory-wide `y_min`/`y_max` computed once before the loop with a running min/max recomputed at each step `t` from `self.trials[:t+1]` (i.e., causal — only trials up to and including the current one), used to quantize `self.trials[t].metric`. Concretely: iterate trials in order, maintain running `y_min`/`y_max` updated *before* quantizing each trial's own metric (so the current trial's value can inform its own scale — matching what will be true at inference the instant it completes), and quantize with that running pair.

This makes `get_prompt()` behave identically whether called once on a full trajectory (training data generation) or incrementally as trials arrive (`optformer_searcher.py`'s `on_trial_complete` → `add_trial` → next `suggest()` → `get_prompt()`), because the token for trial `t` never depends on trials `>t`.

### 2. Asymmetric percentile-based robust range (outlier protection)

Add an optional, **asymmetric** percentile-clip step when computing the (now-running, causal) min/max: since `History` always internally treats lower metric = better (negation already applied upstream for `mode == 'max'` in `load_data.py:177-178`), only the upper ("bad") tail is clipped by default:

- New `History` fields: `metric_percentile_hi=95.0` (clip `y_max` to this percentile of observed metrics) and `metric_percentile_lo=0.0` (clip `y_min` to this percentile; `0.0` means "true min, no clip" — preserves current behavior on the good side).
- In `get_prompt()`, replace raw `max(...)` with `np.percentile(observed_metrics, metric_percentile_hi)` and raw `min(...)` with `np.percentile(observed_metrics, metric_percentile_lo)` (which is just the true min when `metric_percentile_lo=0.0`).
- This composes with the causal fix: percentiles are computed over the same causal prefix of trials at each step, so the clip bound itself only ever depends on past/current trials.
- Values beyond the clipped bound saturate to token 0 or `q-1` via the existing `quantize()` clamp-free formula (need to confirm `quantize()` produces values within `[0, q-1]` even when `x` is outside `[x_min, x_max]` — currently it does not clamp, so an explicit `np.clip(x, x_min, x_max)` before calling `quantize()` is needed for out-of-range metrics once we deliberately allow `y_min`/`y_max` to be tighter than the true range).

Default values (`hi=95.0`, `lo=0.0`) mean: bad-tail clipping is on by default (matching the fcnet motivation), good-tail clipping is off by default (preserving full resolution for best-found configs). Both are tunable per-experiment if a benchmark needs symmetric or fully-disabled clipping (`hi=100.0` to disable entirely).

Guard for small trial counts (percentile over 1–2 trials degenerates to min/max anyway — fine, no special-casing needed since `np.percentile` handles it gracefully).

### Files to change

- `open_optformer/history.py`: `get_prompt()` (causal running min/max + optional percentile clip), and add the two new `History` dataclass fields with defaults that preserve current behavior unless explicitly enabled.
- `tst/test_history.py`: extend `test_history()` (or add new tests) to cover:
  - Causal behavior: a 3-trial history where trial 2's metric would change trial 0/1's token under the old (non-causal) scheme but must not under the new one.
  - Asymmetric percentile clipping: a trial set with one extreme bad (high-metric) outlier, verifying non-outlier trials get better resolution and the outlier saturates to `q-1`; and a separate case with an extreme good (low-metric) outlier, verifying it is **not** clipped by default (still gets its own true-min-based token, no saturation to 0) since `metric_percentile_lo=0.0` by default.
  - Backward-compat: `metric_percentile_hi=100.0, metric_percentile_lo=0.0` reproduces today's `test_history()` expected output exactly (fully disabled clipping).

No changes needed to `generate_training_data/load_data.py` or `optformer_searcher.py` — both already call `get_prompt()` at the right points; the fix is entirely inside `History.get_prompt()`'s internal min/max computation, so causality falls out automatically for both training-data generation (called once, but now computed causally per-step internally) and inference (called incrementally, now consistent with training).

### Verification

- Run `pytest tst/test_history.py -v` — confirm existing tests pass (or are updated per above) and new causal/percentile tests pass.
- Regenerate a small sample of training strings via `generate_training_data/load_data.py` on one fcnet experiment before/after, and diff the emitted value tokens — confirm non-outlier trials get materially better spread post-fix, and confirm no token for trial `t` changes when a later trial `t' > t` is added/removed from the tail of the trajectory (causality check).
- Optionally, ablate: run the two changes independently (causal-only, percentile-only, both) through a short training/eval cycle if training infra is available, to attribute performance deltas to each fix separately, consistent with the scientific evaluation above.
