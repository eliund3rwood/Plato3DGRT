#!/bin/bash
# scripts/vsd_syn_val.sbatch — Phase-3 VSD on the VAL synthetic scene.
#
#   cd /home/tzofi/orcd/scratch/eli/plato3dgrt/PlatoNeRF-main
#   sbatch scripts/vsd_syn_val.sbatch
#
# Submit from inside PlatoNeRF-main: #SBATCH --output is a relative path
# resolved against the SUBMISSION directory, not against wherever the script
# body later cd's to. That has bitten this project twice.
#
# The train scene has its own script rather than being a $1 arm, so the two runs
# are two independent jobs on two nodes with no shared state to get wrong.
#
# --expname is geom_val_vsd, seeded from the converter's pristine 035000.tar,
# NOT geom_val -- that directory holds the 40-iteration smoke test's
# 035040.tar, and with --ft_path omitted the driver auto-discovers the NEWEST
# checkpoint. Resuming from the smoke test would inherit its albedo drift and,
# worse, restart with t_max_frac already annealed to 0.50, so the noise
# schedule would never run its real arc. That is exactly the "banked progress"
# contamination that made run1-run4's schedules meaningless.

#SBATCH --job-name=vsd_syn_val
#SBATCH --partition=mit_preemptable
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# ~1.59 s/it measured over the smoke test x 20000 iters = ~8.8 h. 12:00 keeps
# this to ONE allocation with margin rather than two with a queue gap.
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/vsd_syn_%x_%A.log

set -euo pipefail

ROOT=/home/tzofi/orcd/scratch/eli/plato3dgrt
GT=$ROOT/geom_test
VA=val_chair_20ec9403d616f96ee6cfc789522bfbab

module load miniforge
# conda-forge's compiler activation hooks use unbound variables and abort under
# set -u, so drop it just around the activate.
set +u
conda activate 3dgrut
set -u
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd $ROOT/PlatoNeRF-main

# No --ft_path: the driver picks the newest checkpoint in logs/<expname>/, so a
# requeue after preemption continues from wherever this run actually got to
# instead of restarting at the seed.
python src/run_vsd_synthetic.py \
  --expname geom_val_vsd \
  --cameras $GT/${VA}_cameras.json \
  --vsd_ref_image data/geom_test/val_chair_ref.png \
  --vsd_ref_index 0 \
  --vsd_checkpoint $ROOT/custom_controlnet/final.pt \
  --vsd_arch v7 \
  --vsd_iters 20000 \
  --i_weights 500 \
  --i_print 100
