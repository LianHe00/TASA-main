EXP_DIR=$1
SEED=$2

if [ -z "$SEED" ]
then
    SEED=2023
fi

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python pipeline/step8_3d_training/eval_no_diff.py hydra/job_logging=none hydra/hydra_logging=none \
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
            task.evaluator.num_k_samples=320
