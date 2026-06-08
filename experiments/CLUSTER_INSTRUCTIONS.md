# Cluster Job Instructions — Confident Splatting Resubmission

Everything below runs from the project root on the cluster.
The cluster has no internet; datasets are pre-staged to scratch on the login node.

---

## 0. Verify Dataset

```bash
ls /scratch/amirrz/datasets/mipnerf360/360_v2/
# Expected: bicycle  bonsai  counter  flowers  garden  kitchen  room  stump  treehill
```

---

## 1. Submit Training Array (28 jobs)

```bash
cd /path/to/Confident-Splatting

TRAIN_JOB=$(sbatch \
  --array=0-27 \
  --parsable \
  MANIFEST=experiments/review_manifest.tsv \
  scripts/slurm/train_review_array.sh)
echo "Train job: $TRAIN_JOB"
```

This runs 28 independent training jobs (each ~30k steps on one A100).
Expected wall-time per job: 8-12 hours for MCMC, 6-10 hours for default.

To check progress:
```bash
squeue -u $USER --format="%.10i %.15j %.8T %.10M %.6D %R"
```

---

## 2. Submit Evaluation Array (28 jobs, after training)

```bash
EVAL_JOB=$(sbatch \
  --array=0-27 \
  --parsable \
  --dependency=afterok:${TRAIN_JOB} \
  MANIFEST=experiments/review_manifest.tsv \
  scripts/slurm/eval_review_array.sh)
echo "Eval job: $EVAL_JOB"
```

Each eval job loads the final checkpoint and runs:
- Validation metrics (PSNR / SSIM / LPIPS)
- Confidence-pruned eval (`val_compressed`)
- Pruning curve sweep (threshold 0→0.9, budget 5%→100%)
- Saves plots to `result_dir/plots/`

---

## 3. Measure Compute Overhead (optional, run after training)

To get wall-clock time for each job:
```bash
sacct -j ${TRAIN_JOB} --format=JobID,JobName,Elapsed,MaxRSS,State -n
```

Compare `Elapsed` for `mcmc_vanilla` vs `mcmc_full_beta` on the same scene.
Use these numbers to fill in the "Training overhead" row in the paper.

---

## 4. Aggregate Results (run on login node after eval completes)

```bash
cd /path/to/Confident-Splatting

python3 scripts/aggregate_review_results.py \
  --results-root /scratch/amirrz/results/cs_resubmit \
  --output-dir /scratch/amirrz/results/cs_resubmit/aggregate
```

Output files for the paper:
| File | Used for |
|---|---|
| `final_metrics_summary.csv` | Main ablation table |
| `budget_summary.csv` | PSNR-vs-fraction curves |
| `ablation_metrics.png` | Bar chart figure |
| `budget_psnr_summary.png` | Compression curve figure |
| `all_pruning_pearson.csv` | ACS correlation table |

---

## 5. What the Results Should Show

For the paper to be accepted, the key checks are:

1. **Ablation table** (garden + kitchen + stump, averaged):
   - `full_beta` should have the best PSNR at 10% retained
   - `no_sparsity`, `no_entropy`, `no_ranking` each drop ≥0.2 dB at 10% retained
   - `scalar_conf` should lag behind `full_beta` at 10% retained — validates Beta

2. **Budget curve** (confidence vs opacity vs random):
   - `confidence` selector should dominate `opacity` and `random` at 10-25% retained
   - If `confidence` matches `opacity` at all budgets, the core claim is in trouble

3. **Threshold sensitivity**:
   - PSNR should be stable (within ±0.5 dB) for thresholds 0.0–0.5
   - Answers the "heuristic threshold" objection

4. **Full Mip-NeRF 360 coverage**:
   - Results across all 9 scenes → move per-scene table to supplement
   - Report averaged metrics in main text

---

## Manifest Summary (28 runs)

| Group | Scenes | Variants | Config | Runs |
|---|---|---|---|---|
| Ablation | garden, kitchen, stump | vanilla, full_beta, no_sparsity, no_entropy, no_ranking, scalar_conf | mcmc | 18 |
| Full coverage | bicycle, treehill, flowers, room, counter, bonsai | full_beta | mcmc | 6 |
| Generality | garden, kitchen | vanilla, full_beta | default | 4 |
| **Total** | | | | **28** |
