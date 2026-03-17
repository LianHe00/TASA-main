import os, glob, hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from natsort import natsorted
from dataset.AffordanceDataset import AffordanceDataset
from dataset.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, time_str
from utils.training import load_ckpt
from torch.utils.data import DataLoader
from utils.evaluator import Segment3DEvaluator

def test(cfg: DictConfig) -> None:
    """ Begin testing with this function

    Args:
        cfg: configuration dict
    """
    test_dir = os.path.join(cfg.eval_dir, 'test-' + time_str(Y=False))
    mkdir_if_not_exists(test_dir)
    viz_dir = os.path.join(test_dir, 'viz')
    mkdir_if_not_exists(viz_dir)
    logger.add(os.path.join(test_dir, 'test.log'))
    logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
    logger.info('[Test] ==> Beign testing..')

    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    # prepare testing dataset
    test_dataset = AffordanceDataset(
        root_dir='data',
        split='val',
        use_processed_data=False,
        use_division=False,
        use_processed_data_3=False,
        use_sam2=True,
        use_sam2_1=False
    )
    logger.info(f'Load test dataset size: {len(test_dataset)}')

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.task.test.batch_size,
        # batch_size=1,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.test.num_workers,
        # num_workers=1,
        # pin_memory=True,
        shuffle=True,
    )

    ## create model and optimizer
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)

    ## load checkpoint
    ckpts = natsorted(glob.glob(os.path.join(cfg.exp_dir, 'ckpt', 'mask_refinement_model*.pt')))
    assert len(ckpts) > 0, 'No checkpoint found.'
    load_ckpt(model, ckpts[-1], map_location=device)
    logger.info(f'Load checkpoint from {ckpts[-1]}')

    # load_ckpt(model, ckpts[-1], map_location=device)
    # ckpt = os.path.join(cfg.exp_dir, 'ckpt', 'mask_refinement_model085000.pt')
    # load_ckpt(model, ckpt, map_location=device)
    # logger.info(f'Load checkpoint from {ckpt}')



    ## create evaluator
    exp_tag = f"{cfg.eval_dir}"
    evaluator = Segment3DEvaluator(exp_tag=exp_tag, viz_dir=viz_dir)

    ## sample
    model.eval()

    # B = test_dataloader.batch_size
    B = 1
    sample_list = []

    for i, data in enumerate(test_dataloader):
        logger.info(f"batch index: {i}, case desc_id: {data['c_desc_id']}")

        # prepare data

        x = data['pred_mask_local'].to(device)  # [B, N]
        x = x.unsqueeze(-1)  # [B, N, 1] - 增加最后一维

        x_kwargs = {}        
        for key in data:
            if key.startswith('c_') :
                if torch.is_tensor(data[key]):
                    x_kwargs[key] = data[key].to(device)
                else:
                    x_kwargs[key] = data[key]

        # gt_mask = data['gt_mask_local'].to(device)  # [B, N]
                    
        
        # 模型前向传播
        with torch.no_grad():
            pred_mask = model(x, **x_kwargs)  # [B, N, 1]
        pred_mask = pred_mask.squeeze(-1)  # [B, N]
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask > 0.5).float()

        for bsi in range(len(pred_mask)):
            res_dict = {}
            for key in data:
                if torch.is_tensor(data[key]):
                    res_dict[key] = data[key][bsi].to(device)
                else:
                    res_dict[key] = data[key][bsi]
            
            # full_pcd = np.concatenate([res_dict['c_pc_xyz'], res_dict['c_pc_feat']], axis=1)
            evaluator.register([res_dict['c_visit_id']], [res_dict['c_desc_id']], res_dict['gt_mask_local'].squeeze(), pred_mask[bsi].squeeze(), "", device)

        ## stop evaluation if reach the max number of samples
        if i + 1 >= cfg.task.evaluator.eval_nbatch:
            break

    print(evaluator.get_latex_str())

    with open(os.path.join(test_dir, 'results.json'), 'w') as f:
        evaluator.save(f)

    logger.info(f'Save results to {os.path.join(test_dir, "results.json")}')


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """ Main function

    Args:
        cfg: configuration dict
    """
    ## setup random seed
    SEED = cfg.seed
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    ## set output logger
    mkdir_if_not_exists(cfg.log_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)
    mkdir_if_not_exists(cfg.eval_dir)

    test(cfg) # testing portal


if __name__ == '__main__':
    import torch
    import random
    import numpy as np
    
    main()