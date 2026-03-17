EXP_NAME=$1

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train_no_diff.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_name=${EXP_NAME} \
            output_dir=outputs \
            platform=TensorBoard \
            diffusion.steps=500 \
            task=contact_gen \
            task.train.batch_size=64 \
            task.train.max_steps=200000 \
            task.train.save_every_step=5000 \
            task.train.phase=train \
            task.dataset.sigma=0.8 \
            task.dataset.sets=["HUMANISE"] \
            task.dataset.num_points=8192 \
            model=cdm \
            model.arch=Perceiver

# bash scripts/train.sh perceiver_division_8192_pointtransformer_with_new_loss_2