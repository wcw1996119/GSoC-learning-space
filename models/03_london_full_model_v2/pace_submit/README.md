# PACE submission scripts

## What this is for

Run STGNN encoder training on UoM PACE HPC for **200 epochs × 3 seeds** in parallel.
This is the next-priority improvement identified in the AI/ML expert critique.

## What you get

- 3 trained STGNN checkpoints (one per seed) in `ckpt/`
- Per-seed training logs in `logs/`
- After training, run the local v2.2 utility ablation again with each STGNN to assess
  Stage 1 contribution to final CPC

## Submission steps

1. **Sync v2 folder to PACE**:
   ```bash
   rsync -avz --exclude data/raw --exclude best_model.pt \
     "D:/GIT/mesa Gsoc/GSoC-learning-space/models/03_london_full_model_v2/" \
     <username>@pace.gatech.edu:/scratch/<username>/v2_london/
   ```
   (Adjust hostname / path for your PACE access.)

2. **Set up Python env on PACE** (one-time):
   ```bash
   module load python/3.11 cuda/12.1
   pip install --user torch torch-geometric pandas numpy geopandas matplotlib scikit-learn
   ```

3. **Submit array job**:
   ```bash
   cd /scratch/<username>/v2_london
   sbatch pace_submit/submit_stgnn_200ep.slurm
   ```
   This launches 3 array tasks (seeds 42, 123, 2024) in parallel on 3 GPUs.

4. **Monitor**:
   ```bash
   squeue -u <username>
   tail -f logs/stgnn_seed42_*.out
   ```

5. **Sync results back**:
   ```bash
   rsync -avz <username>@pace.gatech.edu:/scratch/<username>/v2_london/ckpt/ \
     "D:/GIT/mesa Gsoc/GSoC-learning-space/models/03_london_full_model_v2/ckpt/"
   ```

## Expected runtime

- V100 (16 GB): ~5-8 min per seed × 200 epochs
- Total wall time: ~8 min if all 3 array tasks run in parallel
- If queued: ≤ 1 hour typically on PACE

## After PACE completes

Re-run the ablation with the new STGNN baselines:

```bash
# On local: copy each ckpt/best_model_seed{S}.pt to best_model.pt one at a time,
# regenerate cache, then re-run ablations
for S in 42 123 2024; do
  cp ckpt/best_model_seed${S}.pt best_model.pt
  python data/scripts/cache_baseline.py
  python run_ablations.py  # produces ablation_results_stgnn_seed${S}.csv
done
```

This gives you the **full 3 × 4 × 3 = 36 trial** ablation matrix
(STGNN seeds × utility configs × utility seeds).

## Why this matters

Currently the utility ablation runs against a single 30-epoch STGNN checkpoint.
After PACE training, each utility config is calibrated against a 200-epoch STGNN
trained with multiple seeds, giving **statistically meaningful CPC reporting**
(mean ± std across both encoder and utility seeds).

This addresses the #1 AI/ML expert critique: "30 epochs isn't enough; no multi-seed".
