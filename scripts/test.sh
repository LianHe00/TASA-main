EXP_DIR=$1
SEED=$2

if [ -z "$SEED" ]
then
    SEED=2023
fi

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python test_no_diff.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${EXP_DIR} \
            seed=${SEED} \
            output_dir=outputs \
            diffusion.steps=500 \
            task=contact_gen \
            model=cdm \
            model.arch=Perceiver \
            task.dataset.num_points=8192 \
            task.dataset.sigma=0.8 \
            task.dataset.sets=["HUMANISE"] \
            task.evaluator.k_samples=0 \
            task.evaluator.eval_nbatch=32 \
            task.evaluator.num_k_samples=320\
            # gpu=5

# bash scripts/test.sh ./outputs/2025-08-01_21-40-01_perceiver_division_8192_pointtransformer_with_new_loss_1/ 2023
