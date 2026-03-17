import torch.nn as nn
from typing import Dict, List
from omegaconf import DictConfig
from models.diffusion import gaussian_diffusion as gd
from models.diffusion.respace import SpacedDiffusion, space_timesteps
from utils.registry import Registry

Model = Registry('model')

def create_model(cfg: DictConfig, *args, **kwargs) -> nn.Module:
    """ Create model according to the configuration

    Args:
        cfg: configuration dict
    
    Return:
        Model for prediction
    """
    return Model.get(cfg.model.name)(cfg.model, *args, **kwargs)

# 创建高斯扩散模型
def create_gaussian_diffusion(cfg: DictConfig, *args, **kwargs):
    """ Create gaussian diffusion

    Args:
        cfg: configuration dict
    
    Return:
        Diffusion model
    """

    cfg = cfg.diffusion
    steps = cfg.steps

    # 如果timestep_respacing为空，则设置为steps
    if not cfg.timestep_respacing:
        timestep_respacing = [steps]
    else:
        timestep_respacing = cfg.timestep_respacing
    # 获取高斯扩散的beta值
    betas = gd.get_named_beta_schedule(cfg.noise_schedule, steps)

    # 如果predict_xstart为false，则设置为epsilon
    if not cfg.predict_xstart:
        model_type = gd.ModelMeanType.EPSILON
    else:
        model_type = gd.ModelMeanType.START_X
    # 如果loss_type为MSE，则设置为MSE
    if cfg.loss_type == 'MSE':
        loss_type = gd.LossType.MSE
    # 如果loss_type为RESCALED_MSE，则设置为RESCALED_MSE
    elif cfg.loss_type == 'RESCALED_MSE':
        loss_type = gd.LossType.RESCALED_MSE
    # 如果loss_type为KL，则设置为KL
    elif cfg.loss_type == 'KL':
        loss_type = gd.LossType.KL
    # 如果loss_type为RESCALED_KL，则设置为RESCALED_KL
    elif cfg.loss_type == 'RESCALED_KL':
        loss_type = gd.LossType.RESCALED_KL
    # 如果loss_type为RESCALED_MSE，则设置为RESCALED_MSE
    elif cfg.loss_type == 'RESCALED_MSE':
        loss_type = gd.LossType.RESCALED_MSE

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type= model_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not cfg.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=cfg.rescale_timesteps,
    )

def create_model_and_diffusion(cfg: DictConfig, *args, **kwargs) -> nn.Module:
    """ Create model and diffusion according to the configuration

    Args:
        cfg: configuration dict
    
    Return:
        model and diffusion
    """
    model = create_model(cfg, *args, **kwargs)
    diffusion = create_gaussian_diffusion(cfg, *args, **kwargs)
    return model, diffusion

