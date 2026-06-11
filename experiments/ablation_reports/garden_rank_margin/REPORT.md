# Comprehensive Ablation Report

Runs included: 6

Key files:

- `comprehensive_ablation_summary.csv`: one-row-per-variant metric summary
- `comprehensive_ablation_summary.md`: markdown version for quick paste
- `confidence_stats.csv`: checkpoint-derived confidence distribution stats
- `budget_confidence_rows.csv`: matched-budget confidence pruning rows
- `selector_gain_table.csv`: confidence-vs-opacity/random PSNR deltas
- `threshold_confidence_rows.csv`: threshold sensitivity rows

Plots:

- `confidence_budget_psnr.png`
- `confidence_budget_psnr_heatmap.png`
- `confidence_minus_opacity_gain.png`
- `threshold_stability.png`
- `confidence_distribution_summary.png`

## Quick Summary

| variant | auc_psnr_confidence | psnr_at_keep_25 | psnr_at_keep_10 | conf_mean | conf_std |
| --- | --- | --- | --- | --- | --- |
| default_rank_margin_0_00 | 26.3345 | 27.1593 | 20.5417 | 0.0898 | 0.1665 |
| default_rank_margin_0_05 | 26.3339 | 27.1482 | 20.4926 | 0.0902 | 0.1670 |
| default_rank_margin_0_10 | 26.3420 | 27.1670 | 20.4964 | 0.0902 | 0.1669 |
| default_rank_margin_0_25 | 26.3065 | 27.1481 | 20.5178 | 0.0904 | 0.1674 |
| default_rank_margin_0_50 | 26.2893 | 27.1068 | 20.4884 | 0.0916 | 0.1691 |
| default_rank_margin_1_00 | 26.2225 | 27.0622 | 20.2563 | 0.0940 | 0.1758 |
