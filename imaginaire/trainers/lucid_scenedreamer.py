# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import collections
import os

import torch
import torch.nn as nn

from imaginaire.config import Config
# from imaginaire.generators.spade import Generator as SPADEGenerator # Not needed
from imaginaire.losses import (FeatureMatchingLoss, GaussianKLLoss, PerceptualLoss)
# from imaginaire.model_utils.gancraft.loss import GANLoss # Not needed
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.io import get_checkpoint
from imaginaire.utils.misc import split_labels, to_device
# from imaginaire.utils.trainer import ModelAverage, WrappedModel # Not needed
from imaginaire.utils.visualization import tensor2label

from imaginaire.losses.sds import SDSLoss  # Import the SDS Loss


class Trainer(BaseTrainer):
    r"""Initialize LucidDreamer trainer.

    Args:
        cfg (Config): Global configuration.
        net_G (obj): Generator network.
        opt_G (obj): Optimizer for the generator network.
        sch_G (obj): Scheduler for the generator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self,
                 cfg,
                 net_G,
                 opt_G,
                 sch_G,
                 train_data_loader,
                 val_data_loader):
        super(Trainer, self).__init__(cfg, net_G, None, opt_G,
                                      None, sch_G, None,
                                      train_data_loader, val_data_loader)

        # Initialize the SDS loss.  This is the key addition.
        self.sds_loss = SDSLoss(cfg.sds_loss.pretrained_model_name_or_path, device='cuda', eta=cfg.sds_loss.eta)
        self.text_embedding = self.sds_loss.diffusion_model.get_text_embedding(cfg.sds_loss.text_prompt).detach()

    def _init_loss(self, cfg):
        r"""Initialize loss terms.

        Args:
            cfg (obj): Global configuration.
        """
        # # GAN loss and others are not needed for SDS-only training.
        # if hasattr(cfg.trainer.loss_weight, 'gan'):
        #     self.criteria['GAN'] = GANLoss()
        #     self.weights['GAN'] = cfg.trainer.loss_weight.gan

        # Initialize SDS Loss
        self.criteria['SDS'] = self.sds_loss #already in init
        self.weights['SDS'] = cfg.trainer.loss_weight.sds #This will come from the config


    def _start_of_iteration(self, data, current_iteration):
        r"""Model specific custom start of iteration process. We will do two
        things. First, put all the data to GPU. Second, we will resize the
        input so that it becomes multiple of the factor for bug-free
        convolutional operations. This factor is given by the yaml file.
        E.g., base = getattr(self.net_G, 'base', 32)

        Args:
            data (dict): The current batch.
            current_iteration (int): The iteration number of the current batch.
        """
        data = to_device(data, 'cuda')

        # Sample camera poses and perform ray-voxel intersection.
        with torch.no_grad():
            # self.voxel.sample_world(device) # this is done in get batch
            voxel_id, depth2, raydirs, cam_ori_t, _ = self.net_G_module._get_batch(data['images'].size(0), 'cuda') # get_batch from Base3DGenerator
            #voxel_id, depth2, raydirs, cam_ori_t, _ = self.net_G_module._get_batch(data['images'].size(0), 'cuda') # get_batch from Base3DGenerator
        data['voxel_id'] = voxel_id
        data['depth2'] = depth2
        data['raydirs'] = raydirs
        data['cam_ori_t'] = cam_ori_t
        return data

    def gen_forward(self, data):
        r"""Compute the loss for LucidDreamer generator.

        Args:
            data (dict): Training data at the current iteration.
        """
        net_G_output = self.net_G(data) # Call the modified generator forward function

        self._time_before_loss()

        # Compute the SDS loss.  This is the key part.
        sds_loss = self.criteria['SDS'](net_G_output['fake_images'], self.text_embedding)

        self.gen_losses['SDS'] = sds_loss

        total_loss = sds_loss * self.weights['SDS']  # Apply the weight from the config.

        self.gen_losses['total'] = total_loss
        return total_loss

    def dis_forward(self, data):
        r"""Compute the loss for discriminator.  Not needed for pure SDS.

        Args:
            data (dict): Training data at the current iteration.
        """
        return None

    def _get_visualizations(self, data):
        r"""Compute visualization image.

        Args:
            data (dict): The current batch.
        """
        with torch.no_grad():
            label_lengths = self.train_data_loader.dataset.get_label_lengths()
            labels = split_labels(data['label'], label_lengths)

            # Get visualization of the real image and segmentation mask.
            # segmap = tensor2label(labels['seg_maps'], label_lengths['seg_maps'], output_normalized_tensor=True)
            # segmap = torch.cat([x.unsqueeze(0) for x in segmap], 0)

            # Get output from GANcraft model
            net_G_output = self.net_G(data)

            vis_images = [data['images'], net_G_output['fake_images']]

            # if self.cfg.trainer.model_average_config.enabled:
            #     net_G_model_average_output = self.net_G.module.averaged_model(data)
            #     vis_images.append(net_G_model_average_output['fake_images'])
        return vis_images

    def _write_loss_meters(self):  # Changed name to avoid potential conflicts.
        r"""Write all loss values to tensorboard."""
        for update, losses in self.losses.items():
            # update is 'gen_update' or 'dis_update'.
            if update == 'dis_update': #Skip this part since were not doing it.
                continue
            assert update == 'gen_update' or update == 'dis_update'
            for loss_name, loss in losses.items():
                if loss is not None:
                    full_loss_name = update + '/' + loss_name
                    if full_loss_name not in self.meters.keys():
                        # Create a new meter if it doesn't exist.
                        self.meters[full_loss_name] = Meter(
                            full_loss_name, reduce=True)
                    self.meters[full_loss_name].write(loss.item())

    def load_checkpoint(self, cfg, checkpoint_path, resume=None, load_sch=True):
        r"""Load network weights, optimizer parameters, scheduler parameters
        from a checkpoint. Overrides BaseTrainer to remove discriminator loading.

        Args:
            cfg (obj): Global configuration.
            checkpoint_path (str): Path to the checkpoint.
            resume (bool or None): If not ``None``, will determine whether or
            not to load optimizers in addition to network weights.
        """
        if os.path.exists(checkpoint_path):
            # If checkpoint_path exists, we will load its weights to
            # initialize our network.
            if resume is None:
                resume = False
        elif os.path.exists(os.path.join(cfg.logdir, 'latest_checkpoint.txt')):
            # This is for resuming the training from the previously saved
            # checkpoint.
            fn = os.path.join(cfg.logdir, 'latest_checkpoint.txt')
            with open(fn, 'r') as f:
                line = f.read().splitlines()
            checkpoint_path = os.path.join(cfg.logdir, line[0].split(' ')[-1])
            if resume is None:
                resume = True
        else:
            # checkpoint not found and not specified. We will train
            # everything from scratch.
            current_epoch = 0
            current_iteration = 0
            print('No checkpoint found.')
            resume = False
            return resume, current_epoch, current_iteration
        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
        current_epoch = 0
        current_iteration = 0
        if resume:
            self.net_G.load_state_dict(checkpoint['net_G'], strict=self.cfg.trainer.strict_resume)
            if not self.is_inference:
                # self.net_D.load_state_dict(checkpoint['net_D'], strict=self.cfg.trainer.strict_resume) # Removed
                if 'opt_G' in checkpoint:
                    current_epoch = checkpoint['current_epoch']
                    current_iteration = checkpoint['current_iteration']
                    self.opt_G.load_state_dict(checkpoint['opt_G'])
                    # self.opt_D.load_state_dict(checkpoint['opt_D']) # Removed
                    if load_sch:
                        self.sch_G.load_state_dict(checkpoint['sch_G'])
                        # self.sch_D.load_state_dict(checkpoint['sch_D']) # Removed
                    else:
                        if self.cfg.gen_opt.lr_policy.iteration_mode:
                            self.sch_G.last_epoch = current_iteration
                        else:
                            self.sch_G.last_epoch = current_epoch
                        # if self.cfg.dis_opt.lr_policy.iteration_mode:
                        #     self.sch_D.last_epoch = current_iteration
                        # else:
                        #     self.sch_D.last_epoch = current_epoch
                    print('Load from: {}'.format(checkpoint_path))
                else:
                    print('Load network weights only.')
        else:
            try:
                self.net_G.load_state_dict(checkpoint['net_G'], strict=self.cfg.trainer.strict_resume)
                # if 'net_D' in checkpoint:
                #     self.net_D.load_state_dict(checkpoint['net_D'], strict=self.cfg.trainer.strict_resume)
            except Exception:
                if self.cfg.trainer.model_average_config.enabled:
                    net_G_module = self.net_G.module.module
                else:
                    net_G_module = self.net_G.module
                if hasattr(net_G_module, 'load_pretrained_network'):
                    net_G_module.load_pretrained_network(self.net_G, checkpoint['net_G'])
                    print('Load generator weights only.')
                else:
                    raise ValueError('Checkpoint cannot be loaded.')

        print('Done with loading the checkpoint.')
        return resume, current_epoch, current_iteration