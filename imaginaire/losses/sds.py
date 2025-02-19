import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler  # Use Hugging Face Diffusers
from safetensors.torch import load_file


class StableDiffusionWrapper(nn.Module):
    """
    A wrapper for Stable Diffusion models loaded from safetensors,
    specifically designed for use with Score Distillation Sampling (SDS).
    This wrapper provides a simplified interface for interacting with
    the Stable Diffusion model, handling text embedding and noise prediction.

    Args:
        pretrained_model_name_or_path (str): Path or name of the pretrained
            Stable Diffusion model (e.g., "runwayml/stable-diffusion-v1-5").
        device (str): Device to load the model onto ('cuda' or 'cpu').
        use_xformers (bool): If True, use xFormers memory-efficient attention.
    """
    def __init__(self, pretrained_model_name_or_path, device='cuda'):
        super().__init__()
        self.device = device

        # Load the Stable Diffusion pipeline.  Crucially, we use the DDIMScheduler
        # because it's deterministic (given a seed) and efficient for SDS.
        self.pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path,
                                                                torch_dtype=torch.float16,
                                                                safety_checker=None
                                                                ).to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.enable_model_cpu_offload()
        # Optimize if xformers is available.
        self.pipe.enable_xformers_memory_efficient_attention()

        self.num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

    def get_text_embedding(self, prompt: str):
        """
        Computes the text embedding for a given prompt.

        Args:
            prompt (str): The text prompt.

        Returns:
            torch.Tensor: The text embedding.  Shape: (1, 77, 768) or similar.
        """
        # "max_length" is the token length of the prompt.
        text_inputs = self.pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        # text_embeddings = self.pipe.text_encoder(text_input_ids)[0].detach().float() #would need float for older sd
        text_embeddings = self.pipe.text_encoder(text_input_ids)[0] #would need float for older sd
        return text_embeddings

    def predict_noise(self, noisy_image, text_embeddings, t):
        """
        Predicts the noise added to a noisy image given text embeddings and
        a timestep.  This is the core of the SDS loss.

        Args:
            noisy_image (torch.Tensor): The noisy image (x_t in SDS).
                Shape: (B, C, H, W).  Typically (1, 3, 512, 512).
            text_embeddings (torch.Tensor): The text embeddings.
                Shape: (1, 77, 768) or similar.
            t (torch.Tensor): The timesteps.  Shape: (B,)

        Returns:
            torch.Tensor: The predicted noise.  Shape matches noisy_image.
        """

        # Concatenate the noisy image and the text embeddings. The order matters!
        latents = torch.cat([noisy_image, text_embeddings], dim=1)

        # Predict the noise residual.
        with torch.no_grad():
            noise_pred = self.pipe.unet(latents, t).sample

        return noise_pred

    def train_step(self, text_embeddings, pred_rgb,
                   guidance_scale=100,
                   grad_clip=None):
        """
        Performs a single training step using SDS loss.

        Args:
            text_embeddings (torch.Tensor): Embeddings from the text prompt.
            pred_rgb (torch.Tensor): Predicted RGB image (rendered from NeRF).
            guidance_scale (float): Scale for the guidance (CFG).
            grad_clip (float, optional): Gradient clipping value.

        Returns:
            torch.Tensor: The SDS loss.
        """
        B = pred_rgb.shape[0]
        K = torch.randint(self.min_step, self.max_step, (B,), dtype=torch.long, device=self.device)
        w = (1 - self.pipe.scheduler.alphas_cumprod[K]).float()
        w = w / (self.pipe.scheduler.alphas_cumprod[K] ** 0.5) / (1 - self.pipe.scheduler.alphas_cumprod[K])
        # w = self.opt.alphas_cumprod[t] ** 0.5 * (1 - self.opt.alphas_cumprod[t]) ** 0.5 # relative
        w = w[:, None, None, None] # Expand w to match dimension

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(pred_rgb)
            latents = self.pipe.scheduler.add_noise(pred_rgb, noise, K)
            # get latents
            latents = torch.cat([latents] * 2)
            # concat latents and text embeddings
            latent_model_input = torch.cat([latents, text_embeddings.repeat(2, 1, 1)], dim=1)

            # predict the noise residual
            u, t = self.pipe.unet(latent_model_input, K).sample.chunk(2)
            # perform classifier-free guidance
            noise_pred = u + guidance_scale * (t - u)

        # if self.opt.grad_clip is not None:
        #     noise_pred = torch.clamp(noise_pred, -self.opt.grad_clip, self.opt.grad_clip)

        # w (bs, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (pred_rgb - grad).detach()
        loss = 0.5 * F.mse_loss(pred_rgb, target, reduction='sum') / B

        return loss

class SDSLoss(nn.Module):
    """
    The Score Distillation Sampling (SDS) loss.

    Args:
        pretrained_model_name_or_path (str): Path to the pretrained Stable
            Diffusion model.
        device (str): Device to use ('cuda' or 'cpu').
        num_train_timesteps (int): Number of timesteps in the diffusion process.
        eta (float): The η parameter, weight of SDS loss.
    """
    def __init__(self, pretrained_model_name_or_path, device='cuda', eta=1.0):
        super().__init__()
        self.diffusion_model = StableDiffusionWrapper(pretrained_model_name_or_path, device)
        self.eta = eta

    def forward(self, pred_rgb, text_embeddings):
        r"""Computes the SDS loss.

        Args:
            pred_rgb (torch.Tensor): Predicted RGB image from the generator, scaled [-1, 1].
            text_embeddings (torch.Tensor):  Text embeddings for the guiding text.

        Returns:
            torch.Tensor: The SDS loss.
        """
        loss = self.diffusion_model.train_step(text_embeddings, pred_rgb) * self.eta
        return loss