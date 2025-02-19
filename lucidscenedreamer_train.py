import os
import sys
import random
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd.profiler as profiler
import wandb
from imaginaire.config import Config
from imaginaire.utils.cudnn import init_cudnn
from imaginaire.utils.dataset import get_train_and_val_dataloader
from imaginaire.utils.distributed import init_dist, is_master, get_world_size
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.utils.logging import init_logging, make_logging_dir
from imaginaire.utils.trainer import (get_model_optimizer_and_scheduler,
                                      get_trainer, set_random_seed)
import imaginaire.config


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', required=True,
                        help='Path to the training config file.')
    parser.add_argument('--logdir', help='Dir for saving logs and models.')
    parser.add_argument('--checkpoint', default='',
                        help='Checkpoint path.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true',
                        help='Use only one GPU.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (less data, faster training).')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling with PyTorch profiler.')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging.')
    parser.add_argument('--wandb_name', default='default', type=str,
                        help='WandB project name.')
    parser.add_argument('--wandb_id', type=str,
                        help='WandB run ID for resuming.')
    parser.add_argument('--resume', type=int,
                        help='Resume from a specific epoch (if set).')
    parser.add_argument('--num_workers', type=int,
                        help='Override number of data loading workers.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_affinity(args.local_rank)
    set_random_seed(args.seed, by_rank=True)

    cfg = Config(args.config)

    # If single_gpu is set, disable distributed data parallel
    if args.single_gpu:
        cfg.local_rank = args.local_rank = -1
    else:
        cfg.local_rank = args.local_rank
        init_dist(cfg.local_rank)

    print(f"Training with {get_world_size()} GPUs.")

    # Global arguments.
    imaginaire.config.DEBUG = args.debug

    # Override the number of data loading workers if necessary
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers

    # Create log directory for storing training results.
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)
    make_logging_dir(cfg.logdir)

    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    # Initialize data loaders and models.
    batch_size = cfg.data.train.batch_size
    total_step = max(1, cfg.trainer.gen_step)  # Only gen steps in LucidSceneDreamer.
    cfg.data.train.batch_size *= total_step
    train_data_loader, val_data_loader = get_train_and_val_dataloader(cfg, args.seed)

    # Initialize only the generator; no discriminator for SDS-only training.
    net_G, _, opt_G, _, sch_G, _ = \
        get_model_optimizer_and_scheduler(cfg, seed=args.seed)
    trainer = get_trainer(cfg, net_G, None,  # No discriminator.
                          opt_G, None,
                          sch_G, None,
                          train_data_loader, val_data_loader)
    resumed, current_epoch, current_iteration = trainer.load_checkpoint(
        cfg, args.checkpoint, args.resume
    )

    # Initialize Wandb.
    if is_master() and args.wandb:
        if args.wandb_id is not None:
            wandb_id = args.wandb_id
        else:
            if resumed and os.path.exists(os.path.join(cfg.logdir, 'wandb_id.txt')):
                with open(os.path.join(cfg.logdir, 'wandb_id.txt'), 'r+') as f:
                    wandb_id = f.read()
            else:
                wandb_id = wandb.util.generate_id()
                with open(os.path.join(cfg.logdir, 'wandb_id.txt'), 'w+') as f:
                    f.write(wandb_id)
        wandb_mode = "disabled" if args.debug else "online"
        wandb.init(id=wandb_id,
                   project=args.wandb_name,
                   config=cfg,
                   name=os.path.basename(cfg.logdir),
                   resume="allow",
                   settings=wandb.Settings(start_method="fork"),
                   mode=wandb_mode)
        wandb.config.update({'dataset': cfg.data.name})
        wandb.watch(trainer.net_G_module)
        # No discriminator to watch.

    # Start training.
    for epoch in range(current_epoch, cfg.max_epoch):
        print('Epoch {} ...'.format(epoch))
        if not args.single_gpu:
            train_data_loader.sampler.set_epoch(epoch)
        trainer.start_of_epoch(epoch)
        for it, data in enumerate(train_data_loader):
            with profiler.profile(enabled=args.profile,
                                  use_cuda=True,
                                  profile_memory=True,
                                  record_shapes=True) as prof:
                data['prompt'] = cfg.data.train.prompt  # Add the prompt to the data dict.
                data = trainer.start_of_iteration(data, current_iteration)

                # for i in range(cfg.trainer.gen_step): # Only generator steps
                #     trainer.gen_update(
                #         slice_tensor(data, i * batch_size,
                #                      (i + 1) * batch_size))
                # Removed slicing since it is not necessary, can all be one batch
                trainer.gen_update(data)
                current_iteration += 1
                trainer.end_of_iteration(data, epoch, current_iteration)

                if current_iteration >= cfg.max_iter:
                    print('Done with training!!!')
                    return

            if args.profile:
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                prof.export_chrome_trace(os.path.join(cfg.logdir, "trace.json"))

        current_epoch += 1
        trainer.end_of_epoch(data, current_epoch, current_iteration)

    print('Done with training!!!')
    return

if __name__ == "__main__":
    main()