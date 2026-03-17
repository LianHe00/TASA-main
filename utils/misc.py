import torch
import smplkit as sk

def compute_optimization_loss(opt_joints, joints, opt_params=None):
    l, j, _ = joints.shape

    loss = 0.

    ## joints loss
    joints_loss = torch.mean((opt_joints[:, :j, :] - joints) ** 2)
    loss += joints_loss

    ## params loss, smoothness
    if opt_params is not None:
        velocity = opt_params[1:] - opt_params[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        params_loss = torch.mean(acceleration ** 2)
        loss += params_loss * 0.1

    return loss
