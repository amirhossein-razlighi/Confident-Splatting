# Review-Response Experiment Plan

This plan targets the concrete reviewer objections from the previous
submission: missing ablations, heuristic thresholding, limited comparisons,
unclear dataset selection, unclear ACS behavior, and hard-to-read figures.

## Reviewer Complaints to Evidence

1. Ablations are missing.
   - Run the full method and remove one component at a time: sparsity loss,
     Beta entropy, saliency ranking, and Beta parameterization.
   - The key comparison is not just final PSNR, but the quality/compression
     curve after pruning.

2. Thresholding looks heuristic.
   - Report threshold sensitivity curves.
   - Also report matched-budget curves, where confidence, opacity, and random
     pruning keep exactly the same fraction of splats.

3. Comparisons are weak.
   - Include a vanilla 3DGS/MCMC checkpoint and post-hoc opacity pruning.
   - Confidence pruning must beat opacity pruning at matched budgets; otherwise
     the method is only a different threshold knob.

4. The Beta distribution claim is under-supported.
   - Add a scalar-confidence ablation with the same sparsity/ranking losses.
   - If Beta does not improve the curve, the paper should present the method as
     learned confidence pruning rather than emphasize the distributional form.

5. ACS is not validated.
   - Treat ACS as a diagnostic until correlations are measured.
   - Report Pearson/Spearman correlations between average confidence, PSNR,
     SSIM, LPIPS, splat count, and retained fraction across scenes and pruning
     budgets.

6. Dataset choice is under-justified.
   - Run at least one low/medium/high-complexity scene family: object/landmark,
     indoor or bounded scene, and large outdoor 360 scene.
   - Report per-scene and averaged results; move exhaustive per-scene tables to
     supplement.

## Initial Design

Train each selected scene with these variants:

- `vanilla`: no confidence scores.
- `full_beta`: Beta confidence with sparsity, entropy, and ranking losses.
- `no_sparsity`: full method with `lambda_sparsity=0`.
- `no_entropy`: full method with `beta_ent=0`.
- `no_ranking`: full method with `rank_weight=0`.
- `scalar_conf`: learned scalar confidence instead of Beta confidence.

Evaluate each checkpoint with:

- Standard validation metrics: PSNR, SSIM, LPIPS, number of splats, memory,
  render time.
- Threshold curves: confidence threshold vs quality and retained splats.
- Matched-budget curves: keep 100, 75, 50, 35, 25, 15, 10, and 5 percent of
  splats using confidence, opacity, and random scores.
- Qualitative image grids at high, medium, and low budgets.
- ACS/quality correlation reports.

## Tough-Reviewer Critique of This Design

The design is good enough to answer the central rejection reason, but it can
still be attacked in four ways:

1. Opacity pruning is a weak baseline compared with specialized 3DGS
   compression papers. This is acceptable as an internal control, but the paper
   should still compare against published numbers where available.

2. Random pruning is only a sanity check. It should not be framed as a serious
   baseline.

3. Threshold sweeps alone are not enough because users choose thresholds after
   seeing a curve. Matched-budget curves are the fairer main result.

4. ACS correlations across thresholds can be partly confounded by the number of
   retained splats. The paper should report correlations with splat count and
   avoid claiming ACS is a universal quality metric unless the cross-scene
   evidence is strong.

## Final Accepted Design

Keep the training ablations and matched-budget evaluation. In the paper:

- Main table: averaged full-method vs ablations at one or two fixed retained
  budgets, e.g. 25 percent and 10 percent.
- Main curve: PSNR/LPIPS vs retained splat fraction for confidence, opacity,
  and random pruning.
- Supplement: all per-scene threshold curves and qualitative grids.
- ACS claim: report as an empirical diagnostic with correlation table, not as
  an unqualified new quality metric unless the correlations are high and stable.

The current code writes the needed artifacts under each run's `plots/`
directory and aggregates them with `scripts/aggregate_review_results.py`.
