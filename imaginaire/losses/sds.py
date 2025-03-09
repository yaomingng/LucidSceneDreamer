import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler

class SDSLoss(nn.Module):
    def __init__(self, device, pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base",
                 guidance_scale=7.5, t_range=[0.02, 0.98], precision=torch.float32):
        super().__init__()

        self.device = device
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.guidance_scale = guidance_scale
        self.dtype = precision

        pipe = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) 

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred

    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        t = torch.randint(self.min_step, self.max_step,
                         (latents.shape[0],), device=latents.device)
        noise = torch.randn_like(latents)
        latents_noisy = torch.sqrt(
            self.alphas[t]) * latents + torch.sqrt(1 - self.alphas[t]) * noise

        noise_pred = self.get_noise_preds(
            latents_noisy, t, text_embeddings, guidance_scale)
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(
            latents.float(), targets, reduction='sum') / latents.shape[0]
        return loss

    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # Resize to 512x512 if not already that size
        if imgs.shape[2:] != (512, 512):
            imgs = F.interpolate(imgs, size=(512, 512), mode="bilinear", align_corners=False)
        latents = self.vae.encode(imgs).latent_dist.sample() * self.vae.config.scaling_factor

        return latents

    def forward(self, images, text_embeddings):
        latents = self.encode_imgs(images)

        return self.get_sds_loss(latents, text_embeddings, self.guidance_scale)