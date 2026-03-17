# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import os
import functools
import torch
import torch.nn as nn
from loguru import logger

from utils.io import Board
from models.diffusion.resample import uniform_sampling

class TrainLoop:
    def __init__(self, *, cfg, model, diffusion, dataloader, **kwargs) -> None:
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader

        self.lr = cfg.lr
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step
        self.save_every_step = cfg.save_every_step

        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps
        
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0
        self.is_distributed = kwargs['is_distributed'] if 'is_distributed' in kwargs else False

        self.step = 1
        self.resume_step = self._load_and_sync_parameters()

        ## set optimizer
        params = []
        nparams = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
                nparams.append(p.nelement())
                if self.gpu == 0:
                    logger.info(f'Add {n} {p.shape} for optimization.')
        if self.gpu == 0:
            logger.info(f'{len(params)} parameters for optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')
        
        self.optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self.step = self.resume_step + 1
            self._load_optimizer_state()
        
    def _load_and_sync_parameters(self):
        """ Load model from checkpoint if provided for resuming. """
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            return int(filename.replace('.pt', '').replace('model', ''))
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.gpu == 0:
                logger.info(f"Loading model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step
        
    def _load_optimizer_state(self):
        """ Load optimizer state from checkpoint if provided for resuming. """
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint),
            "opt.pt"
        )
        
        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.gpu == 0:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save(self):
        """ Save model and optimizer state. """
        saved_state_dict = {}
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            if 'scene_model' in key or 'clip_model' in key or 'text_model' in key or 'bert_model' in key:
                continue

            saved_state_dict[key] = model_state_dict[key]
        
        with open(os.path.join(self.save_dir, f"model{self.step:06d}.pt"), "wb") as f:
            torch.save(saved_state_dict, f)

        with open(os.path.join(self.save_dir, f"opt.pt"), "wb") as f: # only save the last optimizer state for saving space
            torch.save(self.optimizer.state_dict(), f)
        
        if self.gpu == 0:
            logger.info(f'Model saved! [Step: {self.step:06d}]')
    
    # 冻结场景模型中
    def _freeze_scene_model_batchnorm(self):
        """ Freeze batchnorm in scene model if the model has scene model. """
        if hasattr(self.model, 'scene_model') and self.model.freeze_scene_model :
            for m in self.model.scene_model.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.eval()

    # 训练循环
    def run_loop(self):
        # 遍历每个epoch
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            self._freeze_scene_model_batchnorm() # freeze batchnorm in scene model if the model has scene model
            if self.is_distributed:
                self.dataloader.sampler.set_epoch(epoch)
            # 遍历每个batch
            for it, data in enumerate(self.dataloader): 
                x = data['x'].to(self.device)

                x_kwargs = {}
                if 'x_mask' in data:
                    x_kwargs['x_mask'] = data['x_mask'].to(self.device)
                
                for key in data:
                    if key.startswith('c_') :
                        if torch.is_tensor(data[key]):
                            x_kwargs[key] = data[key].to(self.device)
                        else:
                            x_kwargs[key] = data[key]

                ## one step optimization
                self.optimizer.zero_grad()

                t = uniform_sampling(x.shape[0], self.device, self.diffusion.num_timesteps)
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    x,
                    t,
                    model_kwargs=x_kwargs,
                    epoch=epoch
                )
                terms = compute_losses()
                loss = terms['loss'].mean()
                loss.backward()

                self.optimizer.step()
                self._anneal_lr()
                
                ## log and save
                ## log with loguru, plot with Board
                if self.gpu == 0 and self.step % self.log_every_step == 0:
                    ## log with loguru
                    losses = {key: terms[key].mean().item() for key in terms}

                    logger.info(
                        f"[TRAIN] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | Step: {self.step:7d} | Loss: {losses['loss']:8.5f}"
                    )

                    ## plot with Board
                    write_dict = {'step': self.step, 'train/epoch': epoch}
                    for key in losses:
                        write_dict[f'train/{key}'] = losses[key]
                    Board().write(write_dict)

                if self.gpu == 0 and self.step % self.save_every_step == 0:
                    ## save model
                    self._save()
                
                ## update step and check max steps
                self.step += 1
                if self.step > self.max_steps:
                    return

class CVAETrainLoop:
    def __init__(self, *, cfg, model, dataloader, **kwargs) -> None:
        """ Customized training loop for HUMANISE CVAE
        """
        self.model = model
        self.dataloader = dataloader

        self.lr = cfg.lr
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step
        self.save_every_step = cfg.save_every_step

        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps
        
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0

        self.step = 1
        self.resume_step = self._load_and_sync_parameters()

        ## set optimizer
        tune_params, train_params = [], []
        nparams = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if 'scene_model' in n:
                    tune_params.append(p)
                else:
                    train_params.append(p)
                nparams.append(p.nelement())
                if self.gpu == 0:
                    logger.info(f'Add {n} {p.shape} for optimization.')

        if self.gpu == 0:
            logger.info(f'{len(tune_params) + len(train_params)} parameters for optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')
        
        self.optimizer = torch.optim.Adam(
            [
                {'params': tune_params, 'lr': self.lr * 0.1},
                {'params': train_params}
            ],
            lr=self.lr
        )
        if self.resume_step:
            self.step = self.resume_step + 1
            self._load_optimizer_state()
        
    def _load_and_sync_parameters(self):
        """ Load model from checkpoint if provided for resuming. """
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            return int(filename.replace('.pt', '').replace('model', ''))
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.gpu == 0:
                logger.info(f"Loading model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step
        
    def _load_optimizer_state(self):
        """ Load optimizer state from checkpoint if provided for resuming. """
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint),
            "opt.pt"
        )
        
        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.gpu == 0:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save(self):
        """ Save model and optimizer state. """
        saved_state_dict = {}
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            if 'clip_model' in key or 'text_model' in key or 'bert_model' in key:
                continue

            saved_state_dict[key] = model_state_dict[key]
        
        with open(os.path.join(self.save_dir, f"model{self.step:06d}.pt"), "wb") as f:
            torch.save(saved_state_dict, f)

        with open(os.path.join(self.save_dir, f"opt.pt"), "wb") as f: # only save the last optimizer state for saving space
            torch.save(self.optimizer.state_dict(), f)
        
        if self.gpu == 0:
            logger.info(f'Model saved! [Step: {self.step:06d}]')

    def run_loop(self):
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for it, data in enumerate(self.dataloader): 
                x = data['x'].to(self.device)
                print(x.shape)

                x_kwargs = {}
                if 'x_mask' in data:
                    x_kwargs['x_mask'] = data['x_mask'].to(self.device)
                
                for key in data:
                    if key.startswith('c_') :
                        if torch.is_tensor(data[key]):
                            x_kwargs[key] = data[key].to(self.device)
                        else:
                            x_kwargs[key] = data[key]

                ## one step optimization
                self.optimizer.zero_grad()

                terms = self.model.compute_losses(x, x_kwargs)
                loss = terms['loss'].mean()
                loss.backward()

                self.optimizer.step()
                self._anneal_lr()
                
                ## log and save
                ## log with loguru, plot with Board
                if self.gpu == 0 and self.step % self.log_every_step == 0:
                    ## log with loguru
                    losses = {key: terms[key].mean().item() for key in terms}

                    logger.info(
                        f"[TRAIN] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | Step: {self.step:7d} | Loss: {losses['loss']:8.5f}"
                    )

                    ## plot with Board
                    write_dict = {'step': self.step, 'train/epoch': epoch}
                    for key in losses:
                        write_dict[f'train/{key}'] = losses[key]
                    Board().write(write_dict)

                if self.gpu == 0 and self.step % self.save_every_step == 0:
                    ## save model
                    self._save()
                
                ## update step and check max steps
                self.step += 1
                if self.step > self.max_steps:
                    return

def load_ckpt(model: torch.nn.Module, path: str, map_location='cpu') -> None:
    """ Load checkpoint for model

    Args:
        model: current model
        path: save path
        map_location: 指定加载位置，默认为'cpu'避免GPU内存不足
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    # 使用map_location参数，避免自动加载到GPU
    saved_state_dict = torch.load(path, map_location=map_location)
    model_state_dict = model.state_dict()

    unchanged_weights = []
    used_weights = []
    for key in model_state_dict:
        ## current state and saved state both on single GPU or both on multi GPUs 
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
            logger.info(f'Load parameter {key} for current model.')
            used_weights.append(key)
        
        ## current state on single GPU and saved state on multi GPUs
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
            logger.info(f'Load parameter module.{key} for current model [Trained on multi GPUs].')
            used_weights.append('module.'+key)
        
        if key not in saved_state_dict and 'module.'+key not in saved_state_dict:
            unchanged_weights.append(key)

    unused_weights = []
    for key in saved_state_dict:
        if key not in used_weights:
            unused_weights.append(key)

    for key in unchanged_weights:
        logger.info(f'Unchanged_weight: {key}')
    
    for key in unused_weights:
        logger.info(f'Unused_weight: {key}')
    
    model.load_state_dict(model_state_dict)

class PointCloudMaskRefinementTrainLoop:
    """专门用于点云mask润色的训练循环
    
    特点：
    1. 以点云几何为条件，优化mask质量
    2. 支持渐进式mask优化
    3. 保持点云结构不变，只调整mask标签
    """
    
    def __init__(self, *, cfg, model, diffusion, dataloader, **kwargs) -> None:
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader

        # 训练参数
        self.lr = cfg.lr
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step
        self.save_every_step = cfg.save_every_step

        # 恢复训练
        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps
        
        # 设备配置
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0
        self.is_distributed = kwargs['is_distributed'] if 'is_distributed' in kwargs else False

        # mask润色特定参数
        self.mask_loss_weight = getattr(cfg, 'mask_loss_weight', 1.0)
        self.geometry_loss_weight = getattr(cfg, 'geometry_loss_weight', 0.1)
        self.boundary_loss_weight = getattr(cfg, 'boundary_loss_weight', 0.5)

        self.step = 1
        self.resume_step = self._load_and_sync_parameters()

        # 设置优化器
        params = []
        nparams = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
                nparams.append(p.nelement())
                if self.gpu == 0:
                    logger.info(f'Add {n} {p.shape} for mask refinement optimization.')
        
        if self.gpu == 0:
            logger.info(f'{len(params)} parameters for mask refinement optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')
        
        self.optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        
        if self.resume_step:
            self.step = self.resume_step + 1
            self._load_optimizer_state()
    
    def _load_and_sync_parameters(self):
        """加载模型检查点"""
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            return int(filename.replace('.pt', '').replace('model', ''))
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.gpu == 0:
                logger.info(f"Loading mask refinement model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step
    
    def _load_optimizer_state(self):
        """加载优化器状态"""
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint),
            "opt.pt"
        )
        
        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.gpu == 0:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _anneal_lr(self):
        """学习率衰减"""
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save(self):
        """保存模型和优化器状态"""
        saved_state_dict = {}
        model_state_dict = self.model.state_dict()
        
        # 保存所有权重（mask润色模型通常较小）
        for key in model_state_dict:
            saved_state_dict[key] = model_state_dict[key]
        
        with open(os.path.join(self.save_dir, f"mask_refinement_model{self.step:06d}.pt"), "wb") as f:
            torch.save(saved_state_dict, f)

        with open(os.path.join(self.save_dir, f"mask_refinement_opt.pt"), "wb") as f:
            torch.save(self.optimizer.state_dict(), f)
        
        if self.gpu == 0:
            logger.info(f'Mask refinement model saved! [Step: {self.step:06d}]')

    def _compute_mask_refinement_loss(self, pred_mask, gt_mask, point_cloud, initial_mask):
        """计算mask润色的复合损失
        
        Args:
            pred_mask: 预测的mask [B, N]
            gt_mask: 真实mask [B, N] 
            point_cloud: 点云坐标 [B, N, 3]
            initial_mask: 初始mask [B, N]
        """
        # 1. 基础mask损失
        mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_mask, gt_mask.float()
        )
        
        # 2. 几何一致性损失（保持点云结构）
        # 计算相邻点之间的mask一致性
        geometry_loss = self._compute_geometry_consistency_loss(
            pred_mask, point_cloud
        )
        
        # 3. 边界优化损失
        boundary_loss = self._compute_boundary_loss(
            pred_mask, initial_mask, point_cloud
        )
        
        # 4. 平滑性损失
        smoothness_loss = self._compute_smoothness_loss(pred_mask, point_cloud)
        
        # 组合损失
        total_loss = (
            self.mask_loss_weight * mask_loss +
            self.geometry_loss_weight * geometry_loss +
            self.boundary_loss_weight * boundary_loss +
            0.1 * smoothness_loss
        )
        
        return {
            'loss': total_loss,
            'mask_loss': mask_loss,
            'geometry_loss': geometry_loss,
            'boundary_loss': boundary_loss,
            'smoothness_loss': smoothness_loss
        }
    
    def _compute_geometry_consistency_loss(self, pred_mask, point_cloud):
        """计算几何一致性损失"""
        # 简化的几何一致性：相邻点应该有相似的mask值
        # 这里使用KNN找到最近邻
        batch_size, num_points, _ = point_cloud.shape
        
        # 计算点云距离矩阵（简化版本）
        dist_matrix = torch.cdist(point_cloud, point_cloud)
        
        # 找到每个点的K个最近邻
        K = 8
        _, knn_indices = torch.topk(dist_matrix, k=K+1, dim=-1, largest=False)
        knn_indices = knn_indices[:, :, 1:]  # 排除自己
        
        # 计算mask一致性
        pred_probs = torch.sigmoid(pred_mask)
        consistency_loss = 0
        
        for i in range(batch_size):
            for j in range(num_points):
                neighbor_indices = knn_indices[i, j]
                neighbor_masks = pred_probs[i, neighbor_indices]
                center_mask = pred_probs[i, j]
                
                # 计算与邻居的mask差异
                diff = torch.abs(neighbor_masks - center_mask)
                consistency_loss += torch.mean(diff)
        
        return consistency_loss / (batch_size * num_points)
    
    def _compute_boundary_loss(self, pred_mask, initial_mask, point_cloud):
        """计算边界优化损失"""
        # 找到初始mask的边界点
        pred_probs = torch.sigmoid(pred_mask)
        
        # 简化的边界损失：边界附近的点应该有平滑的mask过渡
        boundary_loss = torch.nn.functional.mse_loss(
            pred_probs, initial_mask.float()
        )
        
        return boundary_loss
    
    def _compute_smoothness_loss(self, pred_mask, point_cloud):
        """计算平滑性损失"""
        # 使用拉普拉斯算子计算平滑性
        pred_probs = torch.sigmoid(pred_mask)
        
        # 简化的平滑性损失
        smoothness_loss = torch.mean(torch.abs(
            pred_probs[:, 1:] - pred_probs[:, :-1]
        ))
        
        return smoothness_loss

    def run_loop(self):
        """mask润色训练循环"""
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            
            if self.is_distributed:
                self.dataloader.sampler.set_epoch(epoch)
            
            for it, data in enumerate(self.dataloader):
                # 数据准备
                point_cloud = data['point_cloud'].to(self.device)  # [B, N, 3]
                initial_mask = data['initial_mask'].to(self.device)  # [B, N]
                gt_mask = data['gt_mask'].to(self.device)  # [B, N]
                
                # 准备条件数据
                x_kwargs = {
                    'point_cloud': point_cloud,
                    'initial_mask': initial_mask
                }
                
                # 优化步骤
                self.optimizer.zero_grad()
                
                # 扩散过程
                t = uniform_sampling(point_cloud.shape[0], self.device, self.diffusion.num_timesteps)
                
                # 计算损失
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    gt_mask,  # 使用gt_mask作为目标
                    t,
                    model_kwargs=x_kwargs,
                    epoch=epoch
                )
                
                terms = compute_losses()
                
                # 添加mask润色特定的损失
                pred_mask = self.model(gt_mask, t, **x_kwargs)
                refinement_losses = self._compute_mask_refinement_loss(
                    pred_mask, gt_mask, point_cloud, initial_mask
                )
                
                # 组合损失
                total_loss = terms['loss'].mean() + 0.1 * refinement_losses['loss']
                
                total_loss.backward()
                self.optimizer.step()
                self._anneal_lr()
                
                # 日志记录
                if self.gpu == 0 and self.step % self.log_every_step == 0:
                    losses = {key: terms[key].mean().item() for key in terms}
                    refinement_losses_dict = {
                        key: refinement_losses[key].item() for key in refinement_losses
                    }
                    
                    logger.info(
                        f"[MASK REFINEMENT] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | "
                        f"Step: {self.step:7d} | Loss: {losses['loss']:8.5f} | "
                        f"Refinement Loss: {refinement_losses_dict['loss']:8.5f}"
                    )
                    
                    # 记录到Board
                    write_dict = {
                        'step': self.step, 
                        'train/epoch': epoch,
                        'train/total_loss': total_loss.item()
                    }
                    
                    for key in losses:
                        write_dict[f'train/{key}'] = losses[key]
                    for key in refinement_losses_dict:
                        write_dict[f'train/refinement_{key}'] = refinement_losses_dict[key]
                    
                    Board().write(write_dict)
                
                # 保存模型
                if self.gpu == 0 and self.step % self.save_every_step == 0:
                    self._save()
                
                # 更新步数
                self.step += 1
                if self.step > self.max_steps:
                    return

class SimpleMaskRefinementTrainLoop:
    """简单的点云mask润色训练循环
    
    特点：
    1. 不使用扩散模型，直接用模型推理
    2. 以点云几何为条件，优化mask质量
    3. 支持多种损失函数组合
    4. 保持点云结构不变，只调整mask标签
    """
    
    def __init__(self, *, cfg, model, dataloader, **kwargs) -> None:
        self.model = model
        self.dataloader = dataloader

        # 训练参数
        self.lr = cfg.lr
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step
        self.save_every_step = cfg.save_every_step

        # 恢复训练
        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps
        
        # 设备配置
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0
        self.is_distributed = kwargs['is_distributed'] if 'is_distributed' in kwargs else False

        # mask润色特定参数
        self.mask_loss_weight = getattr(cfg, 'mask_loss_weight', 1.0)
        self.geometry_loss_weight = getattr(cfg, 'geometry_loss_weight', 0.1)
        self.boundary_loss_weight = getattr(cfg, 'boundary_loss_weight', 0.5)
        self.smoothness_loss_weight = getattr(cfg, 'smoothness_loss_weight', 0.1)
        self.consistency_loss_weight = getattr(cfg, 'consistency_loss_weight', 0.2)

        self.step = 1
        self.resume_step = self._load_and_sync_parameters()

        # 设置优化器
        params = []
        nparams = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
                nparams.append(p.nelement())
                if self.gpu == 0:
                    logger.info(f'Add {n} {p.shape} for mask refinement optimization.')
        
        if self.gpu == 0:
            logger.info(f'{len(params)} parameters for mask refinement optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')
        
        self.optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        
        if self.resume_step:
            self.step = self.resume_step + 1
            self._load_optimizer_state()
    
    def _load_and_sync_parameters(self):
        """加载模型检查点"""
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            return int(filename.replace('.pt', '').replace('model', ''))
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.gpu == 0:
                logger.info(f"Loading mask refinement model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step
    
    def _load_optimizer_state(self):
        """加载优化器状态"""
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint),
            "opt.pt"
        )
        
        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.gpu == 0:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _anneal_lr(self):
        """学习率衰减"""
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save(self):
        """保存模型和优化器状态"""

        saved_state_dict = {}
        model_state_dict = self.model.state_dict()
        
        # 保存所有权重（mask润色模型通常较小）
        for key in model_state_dict:
            if 'clip_model' in key or 'text_model' in key or 'bert_model' in key:
                continue
            saved_state_dict[key] = model_state_dict[key]
        
        with open(os.path.join(self.save_dir, f"mask_refinement_model{self.step:06d}.pt"), "wb") as f:
            torch.save(saved_state_dict, f)

        with open(os.path.join(self.save_dir, f"mask_refinement_opt.pt"), "wb") as f:
            torch.save(self.optimizer.state_dict(), f)
        
        if self.gpu == 0:
            logger.info(f'Mask refinement model saved! [Step: {self.step:06d}]')

    def _compute_mask_refinement_loss(self, pred_mask, gt_mask, point_cloud, initial_mask):
        """计算mask润色的复合损失
        
        Args:
            pred_mask: 预测的mask logits [B, N]
            gt_mask: 真实mask [B, N] 
            point_cloud: 点云坐标 [B, N, 3]
            initial_mask: 初始mask [B, N]
        """
        batch_size, num_points = pred_mask.shape
        
        # 1. 基础mask损失（BCE）
        # mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #     pred_mask, gt_mask.float()
        # )
        
        # # 2. Dice损失（更好的mask质量评估）
        pred_probs = torch.sigmoid(pred_mask)
        dice_loss = self._compute_dice_loss(pred_probs, gt_mask.float())
        
        # # 3. 添加Focal Loss来处理类别不平衡
        # focal_loss = self._compute_focal_loss(pred_mask, gt_mask.float())
        
        # # 4. 添加IoU损失
        # iou_loss = self._compute_iou_loss(pred_probs, gt_mask.float())
        
        # # 5. 添加统计信息用于调试
        stats = self._compute_mask_stats(pred_probs, gt_mask.float())
        
        # 组合损失 - 使用更平衡的权重
        total_loss = (
            # 0.3 * mask_loss +
            # 0.3 * dice_loss +
            # 0.4 * mask_loss +
            dice_loss
            # 0.2 * focal_loss 
            # 0.2 * iou_loss
        )

        # # 组合损失 - 使用更平衡的权重
        # total_loss = (
        #     # 0.3 * mask_loss +
        #     # 0.3 * dice_loss +
        #     0.5 * mask_loss +
        #     0.5 * dice_loss
        #     # 0.2 * focal_loss +
        #     # 0.2 * iou_loss
        # )
        
        return {
            'loss': total_loss,
            'mask_loss': torch.tensor(0.0),
            # 'dice_loss': torch.tensor(0.0),
            'focal_loss': torch.tensor(0.0),
            'iou_loss': torch.tensor(0.0),
            # 'mask_loss': mask_loss,
            'dice_loss': dice_loss,
            # 'focal_loss': focal_loss,
            # 'iou_loss': iou_loss,
            'pred_mean': stats['pred_mean'],
            'gt_mean': stats['gt_mean'],
            'pred_std': stats['pred_std'],
            'gt_std': stats['gt_std'],
            'positive_ratio': stats['positive_ratio']
        }
    
    def _compute_focal_loss(self, pred_logits, gt_mask, alpha=0.25, gamma=2.0):
        """计算Focal Loss来处理类别不平衡"""
        pred_probs = torch.sigmoid(pred_logits)
        
        # 计算focal loss
        pt = pred_probs * gt_mask + (1 - pred_probs) * (1 - gt_mask)
        focal_weight = (1 - pt) ** gamma
        
        # 添加alpha权重
        alpha_weight = alpha * gt_mask + (1 - alpha) * (1 - gt_mask)
        
        focal_loss = -alpha_weight * focal_weight * torch.log(pt + 1e-6)
        return torch.mean(focal_loss)
    
    def _compute_iou_loss(self, pred_probs, gt_mask):
        """计算IoU损失"""
        intersection = torch.sum(pred_probs * gt_mask, dim=1)
        union = torch.sum(pred_probs, dim=1) + torch.sum(gt_mask, dim=1) - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1.0 - torch.mean(iou)
        
        return iou_loss
    
    def _compute_mask_stats(self, pred_probs, gt_mask):
        """计算mask统计信息用于调试"""
        return {
            'pred_mean': torch.mean(pred_probs),
            'gt_mean': torch.mean(gt_mask),
            'pred_std': torch.std(pred_probs),
            'gt_std': torch.std(gt_mask),
            'positive_ratio': torch.mean(gt_mask)  # 正样本比例
        }
    
    def _compute_dice_loss(self, pred_probs, gt_mask):
        """计算Dice损失"""
        # Dice系数 = 2*|A∩B| / (|A|+|B|)
        intersection = torch.sum(pred_probs * gt_mask, dim=1)
        union = torch.sum(pred_probs, dim=1) + torch.sum(gt_mask, dim=1)
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - torch.mean(dice)
        
        return dice_loss

    def run_loop(self):
        """mask润色训练循环"""
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            
            if self.is_distributed:
                self.dataloader.sampler.set_epoch(epoch)
            
            for it, data in enumerate(self.dataloader):
                # 数据准备
                # x = data['x'].to(self.device)  # [B, N]
                x = data['pred_mask_local'].to(self.device)  # [B, N]
                x = x.unsqueeze(-1)  # [B, N, 1] - 增加最后一维

                x_kwargs = {}        
                for key in data:
                    if key.startswith('c_') :
                        if torch.is_tensor(data[key]):
                            x_kwargs[key] = data[key].to(self.device)
                        else:
                            x_kwargs[key] = data[key]

                gt_mask = data['gt_mask_local'].to(self.device)  # [B, N]
                         
                # 优化步骤
                self.optimizer.zero_grad()
                
                # 模型前向传播
                pred_mask = self.model(x, **x_kwargs)  # [B, N, 1]
                pred_mask = pred_mask.squeeze(-1)  # [B, N]
                
                # 计算损失
                losses = self._compute_mask_refinement_loss(
                    pred_mask, gt_mask, x_kwargs['c_pc_xyz'], x.squeeze(-1)
                )
            

                total_loss = losses['loss']
                total_loss.backward()
                self.optimizer.step()
                self._anneal_lr()
                
                # 日志记录
                if self.gpu == 0 and self.step % self.log_every_step == 0:
                    losses_dict = {key: losses[key].item() for key in losses}
                    
                    logger.info(
                        f"[MASK REFINEMENT] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | "
                        f"Step: {self.step:7d} | Loss: {losses_dict['loss']:8.5f} | "
                        f"BCE: {losses_dict['mask_loss']:6.4f} | Dice: {losses_dict['dice_loss']:6.4f} | "
                        f"Focal: {losses_dict['focal_loss']:6.4f} | IoU: {losses_dict['iou_loss']:6.4f} | "
                        f"PosRatio: {losses_dict['positive_ratio']:4.3f} | "
                        f"PredMean: {losses_dict['pred_mean']:4.3f} | GtMean: {losses_dict['gt_mean']:4.3f}"
                    )
                    
                    # 记录到Board
                    write_dict = {
                        'step': self.step, 
                        'train/epoch': epoch,
                        'train/total_loss': total_loss.item()
                    }
                    
                    for key in losses_dict:
                        write_dict[f'train/{key}'] = losses_dict[key]
                    
                    Board().write(write_dict)
                
                if self.step == 1:
                    self._save()

                # 保存模型
                if self.gpu == 0 and self.step % self.save_every_step == 0:
                    self._save()
                
                # 更新步数
                self.step += 1
                if self.step > self.max_steps:
                    return
