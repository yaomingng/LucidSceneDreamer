# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imaginaire.model_utils.gancraft.camctl as camctl
import imaginaire.model_utils.gancraft.mc_utils as mc_utils
import imaginaire.model_utils.gancraft.voxlib as voxlib
from imaginaire.generators.gancraft_base import Base3DGenerator, StyleMLP, SKYMLP, RenderCNN
# from imaginaire.generators.gancraft import StyleEncoder  # No longer needed.
from imaginaire.model_utils.pcg_gen import PCGVoxelGenerator, PCGCache
from imaginaire.utils.distributed import master_only_print as print
from encoding import get_encoder
from imaginaire.model_utils.layers import LightningMLP, ConditionalHashGrid

class Generator(Base3DGenerator):
    r"""LucidSceneDreamer generator. This class extends the Base3DGenerator to
    incorporate style transfer capabilities using Score Distillation Sampling
    (SDS). It leverages a pre-trained Stable Diffusion model for guidance,
    allowing the generation of 3D scenes that match a given text prompt.
    The style encoder is removed, as style information is derived directly from
    the text embeddings and the SDS loss.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
        text_model_dims (int): The size of the output text embeddings.  This
            should match the expected input size of the Stable Diffusion
            model's text encoder.  Defaults to 768, which is common for
            CLIP-based text encoders.
    """

    def __init__(self, gen_cfg, data_cfg, text_model_dims=768):  # Added text_model_dims
        super(Generator, self).__init__(gen_cfg, data_cfg)
        print('LucidSceneDreamer [Hash] on ALL Scenes generator initialization.')

        if gen_cfg.pcg_cache:
            print('[Generator] Loading PCG dataset: ', gen_cfg.pcg_dataset_path)
            self.voxel = PCGCache(gen_cfg.pcg_dataset_path)
            print('[Generator] Loaded PCG dataset.')
        else:
            self.voxel = PCGVoxelGenerator(gen_cfg.scene_size)

        self.blk_feats = None
        self.text_model_dims = text_model_dims # Store this for potential use.

        # Minecraft -> SPADE label translator.
        self.label_trans = mc_utils.MCLabelTranslator()
        self.num_reduced_labels = self.label_trans.get_num_reduced_lbls()
        self.reduced_label_set = getattr(gen_cfg, 'reduced_label_set', False)
        self.use_label_smooth = getattr(gen_cfg, 'use_label_smooth', False)
        self.use_label_smooth_real = getattr(gen_cfg, 'use_label_smooth_real', self.use_label_smooth)
        self.use_label_smooth_pgt = getattr(gen_cfg, 'use_label_smooth_pgt', False)
        self.label_smooth_dia = getattr(gen_cfg, 'label_smooth_dia', 11)

        # Load MLP model.
        self.hash_encoder, self.hash_in_dim = get_encoder(encoding='hashgrid', input_dim=5,
                                                          desired_resolution=2048 * 1, level_dim=8)
        self.render_net = LightningMLP(self.hash_in_dim, viewdir_dim=self.input_dim_viewdir,
                                       style_dim=self.interm_style_dims, mask_dim=self.num_reduced_labels,
                                       out_channels_s=1, out_channels_c=self.final_feat_dim,
                                       **self.mlp_model_kwargs)
        print(self.hash_encoder)
        self.world_encoder = ConditionalHashGrid()

        # Camera sampler.
        self.camera_sampler_type = getattr(gen_cfg, 'camera_sampler_type', "random")
        assert self.camera_sampler_type in ['random', 'traditional']
        self.camera_min_entropy = getattr(gen_cfg, 'camera_min_entropy', -1)
        self.camera_rej_avg_depth = getattr(gen_cfg, 'camera_rej_avg_depth', -1)
        self.cam_res = gen_cfg.cam_res
        self.crop_size = gen_cfg.crop_size

        print('Done with the LucidSceneDreamer initialization.')

    def custom_init(self):
        r"""Weight initialization."""

        def init_func(m):
            if hasattr(m, 'weight'):
                try:
                    nn.init.kaiming_normal_(m.weight.data, a=0.2, nonlinearity='leaky_relu')
                except:
                    print(m.name)
                m.weight.data *= 0.5
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.fill_(0.0)

        self.apply(init_func)

    def _get_batch(self, batch_size, device):
        r"""Sample camera poses and perform ray-voxel intersection.

        Args:
            batch_size (int): Expected batch size of the current batch
            device (torch.device): Device on which the tensors should be stored
        """
        with torch.no_grad():
            self.voxel.sample_world(device)
            voxel_id_batch = []
            depth2_batch = []
            raydirs_batch = []
            cam_ori_t_batch = []
            for b in range(batch_size):
                while True:  # Rejection sampling.
                    # Sample camera pose.
                    if self.camera_sampler_type == 'random':
                        cam_res = self.cam_res
                        cam_ori_t, cam_dir_t, cam_up_t = camctl.rand_camera_pose_thridperson2(self.voxel)
                        # ~24mm fov horizontal.
                        cam_f = 0.5 / np.tan(np.deg2rad(73 / 2) * (np.random.rand(1) * 0.5 + 0.5)) * (cam_res[1] - 1)
                        cam_c = [(cam_res[0] - 1) / 2, (cam_res[1] - 1) / 2]
                        cam_res_crop = [self.crop_size[0] + self.pad, self.crop_size[1] + self.pad]
                        cam_c = mc_utils.rand_crop(cam_c, cam_res, cam_res_crop)
                    elif self.camera_sampler_type == 'traditional':
                        cam_res = self.cam_res
                        cam_c = [(cam_res[0] - 1) / 2, (cam_res[1] - 1) / 2]
                        dice = torch.rand(1).item()
                        if dice > 0.5:
                            cam_ori_t, cam_dir_t, cam_up_t, cam_f = \
                                camctl.rand_camera_pose_tour(self.voxel)
                            cam_f = cam_f * (cam_res[1] - 1)
                        else:
                            cam_ori_t, cam_dir_t, cam_up_t = \
                                camctl.rand_camera_pose_thridperson2(self.voxel)
                            # ~24mm fov horizontal.
                            cam_f = 0.5 / np.tan(np.deg2rad(73 / 2) * (np.random.rand(1) * 0.5 + 0.5)) * (
                                        cam_res[1] - 1)

                        cam_res_crop = [self.crop_size[0] + self.pad, self.crop_size[1] + self.pad]
                        cam_c = mc_utils.rand_crop(cam_c, cam_res, cam_res_crop)
                    else:
                        raise NotImplementedError(
                            'Unknown self.camera_sampler_type: {}'.format(self.camera_sampler_type))

                    # Run ray-voxel intersection test
                    voxel_id, depth2, raydirs = voxlib.ray_voxel_intersection_perspective(
                        self.voxel.voxel_t, cam_ori_t, cam_dir_t, cam_up_t, cam_f, cam_c, cam_res_crop,
                        self.num_blocks_early_stop)

                    if self.camera_rej_avg_depth > 0:
                        depth_map = depth2[0, :, :, 0, :]
                        avg_depth = torch.mean(depth_map[~torch.isnan(depth_map)])
                        if avg_depth < self.camera_rej_avg_depth:
                            continue

                    # Reject low entropy.
                    if self.camera_min_entropy > 0:
                        # Check entropy.
                        maskcnt = torch.bincount(
                            torch.flatten(voxel_id[:, :, 0, 0]), weights=None, minlength=680).float() / \
                                  (voxel_id.size(0) * voxel_id.size(1))
                        maskentropy = -torch.sum(maskcnt * torch.log(maskcnt + 1e-10))
                        if maskentropy < self.camera_min_entropy:
                            continue
                    break

                voxel_id_batch.append(voxel_id)
                depth2_batch.append(depth2)
                raydirs_batch.append(raydirs)
                cam_ori_t_batch.append(cam_ori_t)
            voxel_id = torch.stack(voxel_id_batch, dim=0)
            depth2 = torch.stack(depth2_batch, dim=0)
            raydirs = torch.stack(raydirs_batch, dim=0)
            cam_ori_t = torch.stack(cam_ori_t_batch, dim=0).to(device)
            cam_poses = None
        return voxel_id, depth2, raydirs, cam_ori_t, cam_poses

    def sample_camera(self, data):
        r"""Sample camera randomly and precompute everything used by both Gen and Dis.
        This returns a dictionary of tensors that can be used to train both G and D.

        Args:
            data (dict):
                images (N x 3 x H x W tensor) : Real images
                label (N x C2 x H x W tensor) : Segmentation map
        Returns:
            ret (dict):
                voxel_id (N x H x W x max_samples x 1 tensor): IDs of intersected tensors along each ray.
                depth2 (N x 2 x H x W x max_samples x 1 tensor): Depths of entrance and exit points for each ray-voxel
                intersection.
                raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
                cam_ori_t (N x 3 tensor): Camera origins.
                real_masks (N x C3 x H x W tensor): One-hot segmentation map for real images, with translated labels.
                fake_masks (N x C3 x H x W tensor): One-hot segmentation map for sampled camera views.
        """
        device = torch.device('cuda')
        batch_size = data['images'].size(0)
        # ================ Assemble a batch ==================
        # Requires: voxel_id, depth2, raydirs, cam_ori_t.
        voxel_id, depth2, raydirs, cam_ori_t, _ = self._get_batch(batch_size, device)
        ret = {'voxel_id': voxel_id, 'depth2': depth2, 'raydirs': raydirs, 'cam_ori_t': cam_ori_t}

        # =============== Mask translation ================
        real_masks = data['label']
        if self.reduced_label_set:
            # Translate fake mask (directly from mcid).
            # convert unrecognized labels to 'dirt'.
            # N C H W [1 1 80 80]
            reduce_fake_mask = self.label_trans.mc2reduced(
                voxel_id[:, :, :, 0, :].permute(0, 3, 1, 2).long().contiguous()
                , ign2dirt=True)
            reduce_fake_mask_onehot = torch.zeros([
                reduce_fake_mask.size(0), self.num_reduced_labels, reduce_fake_mask.size(2), reduce_fake_mask.size(3)],
                dtype=torch.float, device=device)
            reduce_fake_mask_onehot.scatter_(1, reduce_fake_mask, 1.0)
            fake_masks = reduce_fake_mask_onehot
            if self.pad != 0:
                fake_masks = fake_masks[:, :, self.pad // 2:-self.pad // 2, self.pad // 2:-self.pad // 2]

            # Translate real mask (data['label']), which is onehot.
            real_masks_idx = torch.argmax(real_masks, dim=1, keepdim=True)
            real_masks_idx[real_masks_idx > 182] = 182

            reduced_real_mask = self.label_trans.coco2reduced(real_masks_idx)
            reduced_real_mask_onehot = torch.zeros([
                reduced_real_mask.size(0), self.num_reduced_labels, reduced_real_mask.size(2),
                reduced_real_mask.size(3)], dtype=torch.float, device=device)
            reduced_real_mask_onehot.scatter_(1, reduced_real_mask, 1.0)
            real_masks = reduced_real_mask_onehot

        # Mask smoothing.
        if self.use_label_smooth:
            fake_masks = mc_utils.segmask_smooth(fake_masks, kernel_size=self.label_smooth_dia)
        if self.use_label_smooth_real:
            real_masks = mc_utils.segmask_smooth(real_masks, kernel_size=self.label_smooth_dia)

        ret['real_masks'] = real_masks
        ret['fake_masks'] = fake_masks

        return ret

    def _forward_perpix_sub(self, blk_feats, worldcoord2, raydirs_in, z, mc_masks_onehot=None, global_enc=None):
        r"""Per-pixel rendering forwarding

        Args:
            blk_feats: Deprecated
            worldcoord2 (N x H x W x L x 3 tensor): 3D world coordinates of sampled points. L is number of samples; N is batch size, always 1.
            raydirs_in (N x H x W x 1 x C2 tensor or None): ray direction embeddings.
            z (N x C3 tensor): Intermediate style vectors.
            mc_masks_onehot (N x H x W x L x C4): One-hot segmentation maps.
        Returns:
            net_out_s (N x H x W x L x 1 tensor): Opacities.
            net_out_c (N x H x W x L x C5 tensor): Color embeddings.
        """
        _x, _y, _z = self.voxel.voxel_t.shape
        delimeter = torch.Tensor([_x, _y, _z]).to(worldcoord2)
        normalized_coord = worldcoord2 / delimeter * 2 - 1
        global_enc = global_enc[:, None, None, None, :].repeat(1, normalized_coord.shape[1], normalized_coord.shape[2], normalized_coord.shape[3], 1)
        normalized_coord = torch.cat([normalized_coord, global_enc], dim=-1)
        feature_in = self.hash_encoder(normalized_coord)

        net_out_s, net_out_c = self.render_net(feature_in, raydirs_in, z, mc_masks_onehot)

        if self.raw_noise_std > 0.:
            noise = torch.randn_like(net_out_s) * self.raw_noise_std
            net_out_s = net_out_s + noise

        return net_out_s, net_out_c

    def _forward_perpix(self, blk_feats, voxel_id, depth2, raydirs, cam_ori_t, z, global_enc):
        r"""Sample points along rays, forwarding the per-point MLP and aggregate pixel features

        Args:
            blk_feats (K x C1 tensor): Deprecated
            voxel_id (N x H x W x M x 1 tensor): Voxel ids from ray-voxel intersection test. M: num intersected voxels, why always 6?
            depth2 (N x 2 x H x W x M x 1 tensor): Depths of entrance and exit points for each ray-voxel intersection.
            raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
            cam_ori_t (N x 3 tensor): Camera origins.
            z (N x C3 tensor): Intermediate style vectors.
        """
        # Generate sky_mask; PE transform on ray direction.
        with torch.no_grad():
            raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
            if self.pe_params[2] == 0 and self.pe_params[3] is True:
                raydirs_in = raydirs_in
            elif self.pe_params[2] == 0 and self.pe_params[3] is False:  # Not using raydir at all
                raydirs_in = None
            else:
                raydirs_in = voxlib.positional_encoding(raydirs_in, self.pe_params[2], -1, self.pe_params[3])

            # sky_mask: when True, ray finally hits sky
            sky_mask = voxel_id[:, :, :, [-1], :] == 0
            # sky_only_mask: when True, ray hits nothing but sky
            sky_only_mask = voxel_id[:, :, :, [0], :] == 0

        with torch.no_grad():
            # Random sample points along the ray
            num_samples = self.num_samples + 1
            if self.sample_use_box_boundaries:
                num_samples = self.num_samples - self.num_blocks_early_stop

            # 10 samples per ray + 4 intersections - 2
            rand_depth, new_dists, new_idx = mc_utils.sample_depth_batched(
                depth2, num_samples, deterministic=self.coarse_deterministic_sampling,
                use_box_boundaries=self.sample_use_box_boundaries, sample_depth=self.sample_depth)

            nan_mask = torch.isnan(rand_depth)
            inf_mask = torch.isinf(rand_depth)
            rand_depth[nan_mask | inf_mask] = 0.0

            worldcoord2 = raydirs * rand_depth + cam_ori_t[:, None, None, None, :]

            # Generate per-sample segmentation label
            voxel_id_reduced = self.label_trans.mc2reduced(voxel_id, ign2dirt=True)
            mc_masks = torch.gather(voxel_id_reduced, -2, new_idx)  # B 256 256 N 1
            mc_masks = mc_masks.long()
            mc_masks_onehot = torch.zeros([mc_masks.size(0), mc_masks.size(1), mc_masks.size(
                2), mc_masks.size(3), self.num_reduced_labels], dtype=torch.float, device=voxel_id.device)
            # mc_masks_onehot: [B H W Nlayer 680]
            mc_masks_onehot.scatter_(-1, mc_masks, 1.0)

        net_out_s, net_out_c = self._forward_perpix_sub(blk_feats, worldcoord2, raydirs_in, z, mc_masks_onehot, global_enc)

        # Handle sky
        sky_raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
        sky_raydirs_in = voxlib.positional_encoding(sky_raydirs_in, self.pe_params_sky[0], -1, self.pe_params_sky[1])
        skynet_out_c = self.sky_net(sky_raydirs_in, z)

        # Blending
        weights = mc_utils.volum_rendering_relu(net_out_s, new_dists * self.dists_scale, dim=-2)

        # If a ray exclusively hits the sky (no intersection with the voxels), set its weight to zero.
        weights = weights * torch.logical_not(sky_only_mask).float()
        total_weights_raw = torch.sum(weights, dim=-2, keepdim=True)  # 256 256 1 1
        total_weights = total_weights_raw

        is_gnd = worldcoord2[..., [0]] <= 1.0  # Y X Z, [256, 256, 4, 3], nan < 1.0 == False
        is_gnd = is_gnd.any(dim=-2, keepdim=True)
        nosky_mask = torch.logical_or(torch.logical_not(sky_mask), is_gnd)
        nosky_mask = nosky_mask.float()

        # Avoid sky leakage
        sky_weight = 1.0-total_weights
        if self.keep_sky_out:
            # keep_sky_out_avgpool overrides sky_replace_color
            if self.sky_replace_color is None or self.keep_sky_out_avgpool:
                if self.keep_sky_out_avgpool:
                    if hasattr(self, 'sky_avg'):
                        sky_avg = self.sky_avg
                    else:
                        if self.sky_global_avgpool:
                            sky_avg = torch.mean(skynet_out_c, dim=[1, 2], keepdim=True)
                        else:
                            skynet_out_c_nchw = skynet_out_c.permute(0, 4, 1, 2, 3).squeeze(-1).contiguous()
                            sky_avg = F.avg_pool2d(skynet_out_c_nchw, 31, stride=1, padding=15, count_include_pad=False)
                            sky_avg = sky_avg.permute(0, 2, 3, 1).unsqueeze(-2).contiguous()
                    # print(sky_avg.shape)
                    skynet_out_c = skynet_out_c * (1.0-nosky_mask) + sky_avg*(nosky_mask)
                else:
                    sky_weight = sky_weight * (1.0-nosky_mask)
            else:
                skynet_out_c = skynet_out_c * (1.0-nosky_mask) + self.sky_replace_color*(nosky_mask)

        if self.clip_feat_map is True:  # intermediate feature before blending & CNN
            rgbs = torch.clamp(net_out_c, -1, 1) + 1
            rgbs_sky = torch.clamp(skynet_out_c, -1, 1) + 1
            net_out = torch.sum(weights*rgbs, dim=-2, keepdim=True) + sky_weight * \
                rgbs_sky  # 576, 768, 4, 3 -> 576, 768, 3
            net_out = net_out.squeeze(-2)
            net_out = net_out - 1
        elif self.clip_feat_map is False:
            rgbs = net_out_c
            rgbs_sky = skynet_out_c
            net_out = torch.sum(weights*rgbs, dim=-2, keepdim=True) + sky_weight * \
                rgbs_sky  # 576, 768, 4, 3 -> 576, 768, 3
            net_out = net_out.squeeze(-2)
        elif self.clip_feat_map == 'tanh':
            rgbs = torch.tanh(net_out_c)
            rgbs_sky = torch.tanh(skynet_out_c)
            net_out = torch.sum(weights*rgbs, dim=-2, keepdim=True) + sky_weight * \
                rgbs_sky  # 576, 768, 4, 3 -> 576, 768, 3
            net_out = net_out.squeeze(-2)
        else:
            raise NotImplementedError

        return net_out, new_dists, weights, total_weights_raw, rand_depth, net_out_s, net_out_c, skynet_out_c, \
            nosky_mask, sky_mask, sky_only_mask, new_idx

    def _forward_global(self, net_out, z):
        r"""Forward the CNN

        Args:
            net_out (N x C5 x H x W tensor): Intermediate feature maps.
            z (N x C3 tensor): Intermediate style vectors.

        Returns:
            fake_images (N x 3 x H x W tensor): Output image.
            fake_images_raw (N x 3 x H x W tensor): Output image before TanH.
        """
        fake_images = net_out.permute(0, 3, 1, 2).contiguous()
        fake_images_raw = self.denoiser(fake_images, z)
        fake_images = torch.tanh(fake_images_raw)

        return fake_images, fake_images_raw

    def forward(self, data, random_style=False):
        r"""GANcraft forward with additional text embeddings.
        """
        device = torch.device('cuda')
        batch_size = data['images'].size(0)
        # Requires: voxel_id, depth2, raydirs, cam_ori_t.
        voxel_id, depth2, raydirs, cam_ori_t = data['voxel_id'], data['depth2'], data['raydirs'], data['cam_ori_t']

        global_enc = self.world_encoder(self.voxel.current_height_map, self.voxel.current_semantic_map)

        # Get the text embedding from the data dictionary.  This is where
        # the text prompt's influence comes in.
        text_embedding = data['text_embeddings']

        # ================ Network Forward ================
        # Forward StyleNet
        if self.style_net is not None:
            z = self.style_net(text_embedding)
        else:
            z = text_embedding

        # Forward per-pixel net.
        net_out, new_dists, weights, total_weights_raw, rand_depth, net_out_s, net_out_c, skynet_out_c, nosky_mask, \
        sky_mask, sky_only_mask, new_idx = self._forward_perpix(
            self.blk_feats, voxel_id, depth2, raydirs, cam_ori_t, z, global_enc)

        # Forward global net.
        fake_images, fake_images_raw = self._forward_global(net_out, z)
        if self.pad != 0:
            fake_images = fake_images[:, :, self.pad // 2:-self.pad // 2, self.pad // 2:-self.pad // 2]

        # =============== Arrange Return Values ================
        output = {}
        output['fake_images'] = fake_images

        return output