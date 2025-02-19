# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch

from imaginaire.trainers.gancraft import Trainer as BaseTrainer
from imaginaire.losses.sds import SDSLoss  # Import the SDS loss


class Trainer(BaseTrainer):
    r"""Initialize LucidDreamer trainer.

    Args:
        cfg (Config): Global configuration.
        net_G (obj): Generator network.
        net_D (obj): Discriminator network.
        opt_G (obj): Optimizer for the generator network.
        opt_D (obj): Optimizer for the discriminator network.
        sch_G (obj): Scheduler for the generator optimizer.
        sch_D (obj): Scheduler for the discriminator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """
    def __init__(self, cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
                 train_data_loader, val_data_loader):
        super(Trainer, self).__init__(
            cfg, net_G, net_D, opt_G, opt_D, sch_G, sch_D,
            train_data_loader, val_data_loader)

        # Initialize the SDS loss.  We only need it during training.
        if not self.is_inference:
            self.sds_loss = SDSLoss(
                pretrained_model_name_or_path=cfg.sds_loss.pretrained_model_name_or_path,
                device='cuda'
            )
            self.text_embedding_cache = None  # Optional text embedding cache

    def _init_loss(self, cfg):
        r"""Initialize loss terms.

        Args:
            cfg (obj): Global configuration.
        """
        super()._init_loss(cfg)  # Initialize base class losses.

        # Add SDS loss.
        if hasattr(cfg.trainer.loss_weight, 'sds'):
            # We've already initialized it in __init__.
            self.weights['sds'] = cfg.trainer.loss_weight.sds

    def _start_of_iteration(self, data, current_iteration):
        r"""Adds text embeddings to the data dictionary."""
        # data = super()._start_of_iteration(data, current_iteration) # REMOVE THIS LINE
        data = to_cuda(data) # We still need to move data to cuda.
        # if self.text_embedding_cache is None: # consider using a cached embedding
        with torch.no_grad():
            text_embeddings = self.sds_loss.diffusion_model.get_text_embedding(data['prompt'][0])
            #self.text_embedding_cache = text_embeddings # cache it
        # else:
        #     text_embeddings = self.text_embedding_cache
        data['text_embeddings'] = text_embeddings.repeat(data['voxel_id'].shape[0], 1, 1) # now gets the shape from voxel id instead of image
        return data

    def gen_forward(self, data):
        r"""Compute the loss for the generator.

        Args:
            data (dict): Training data at the current iteration.
        """
        net_G_output = self.net_G(data, random_style=False)

        self._time_before_loss()

        # Compute the standard GAN losses (if needed, using the base class).
        total_loss = super().gen_forward(data)
        # total_loss will be zero if no gan loss 
        if total_loss is None: total_loss = 0

        # Compute SDS loss.
        sds_loss = self.sds_loss(net_G_output['fake_images'], data['text_embeddings'])
        self.gen_losses['sds'] = sds_loss

        total_loss += self.weights['sds'] * sds_loss

        self.gen_losses['total'] = total_loss
        return total_loss