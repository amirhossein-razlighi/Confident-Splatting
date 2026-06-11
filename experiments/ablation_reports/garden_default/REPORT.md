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
| default_full_beta | 26.1675 | 27.0393 | 19.9186 | 0.0981 | 0.1858 |
| default_no_entropy | 26.2675 | 26.9668 | 20.2925 | 0.1035 | 0.1706 |
| default_no_ranking | 26.3137 | 27.1382 | 20.5129 | 0.0900 | 0.1668 |
| default_no_sparsity | 12.9956 | 7.9087 | 7.3588 | 0.8096 | 0.2452 |
| default_scalar_conf | 26.2072 | 26.8907 | 19.9113 | 0.1107 | 0.1739 |
| default_vanilla |  |  |  |  |  |
