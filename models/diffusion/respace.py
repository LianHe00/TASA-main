# This code is based on https://github.com/openai/guided-diffusion
import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion

# 创建时间步列表
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    # 如果section_counts是字符串
    if isinstance(section_counts, str):
        # 如果section_counts以"ddim"开头
        if section_counts.startswith("ddim"):
            # 获取ddimN中的N
            desired_count = int(section_counts[len("ddim") :])
            # 遍历每个时间步
            for i in range(1, num_timesteps):
                # 如果时间步的数量等于desired_count
                if len(range(0, num_timesteps, i)) == desired_count:
                    # 返回时间步列表
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        # 如果section_counts不是字符串
        section_counts = [int(x) for x in section_counts.split(",")]
    # 计算每个section的大小
    size_per = num_timesteps // len(section_counts)
    # 计算剩余的时间步
    extra = num_timesteps % len(section_counts)
    # 初始化时间步列表
    start_idx = 0
    all_steps = []
    # 遍历每个section
    for i, section_count in enumerate(section_counts):
        # 计算当前section的大小
        size = size_per + (1 if i < extra else 0)
        # 如果当前section的大小小于section_count
        if size < section_count:
            # 抛出错误
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        # 如果section_count小于等于1
        if section_count <= 1:
            # 设置步长为1
            frac_stride = 1
        else:
            # 计算步长
            frac_stride = (size - 1) / (section_count - 1)
        # 初始化当前索引
        cur_idx = 0.0
        # 初始化已取时间步列表
        taken_steps = []
        # 遍历每个section
        for _ in range(section_count):
            # 添加当前索引
            taken_steps.append(start_idx + round(cur_idx))
            # 更新当前索引
            cur_idx += frac_stride
        # 添加已取时间步列表
        all_steps += taken_steps
        # 更新起始索引
        start_idx += size
    # 返回时间步列表
    return set(all_steps)

# 创建空间扩散模型
class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """
    # 初始化空间扩散模型
    def __init__(self, use_timesteps, **kwargs):
        # 将use_timesteps转换为集合
        self.use_timesteps = set(use_timesteps)
        # 初始化时间步映射列表
        self.timestep_map = []
        # 获取基础扩散过程的步数
        self.original_num_steps = len(kwargs["betas"])
        # 创建基础扩散过程
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        # 初始化last_alpha_cumprod
        last_alpha_cumprod = 1.0
        # 初始化新的beta列表
        new_betas = []
        # 遍历每个时间步
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            # 如果当前时间步在use_timesteps中
            if i in self.use_timesteps:
                # 添加新的beta
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                # 更新last_alpha_cumprod
                last_alpha_cumprod = alpha_cumprod
                # 添加时间步映射
                self.timestep_map.append(i)
        # 更新betas
        kwargs["betas"] = np.array(new_betas)
        # 初始化基础扩散过程
        super().__init__(**kwargs)

    # 计算均值和方差
    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    # 计算训练损失
    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    # 计算条件均值
    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    # 计算条件得分
    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    # 包装模型
    def _wrap_model(self, model):
        # 如果模型是_WrappedModel类型
        if isinstance(model, _WrappedModel):
            # 返回模型
            return model
        # 返回包装模型
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    # 缩放时间步
    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
