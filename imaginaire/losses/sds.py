import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

class SDSTextEncoder(nn.Module):
    """Wrapper for the Stable Diffusion text encoder."""
    def __init__(self, device, pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder.eval() 
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.device = device
        self.text_encoder = self.text_encoder.to(self.device)

    def forward(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.text_encoder.device))[0]
        return text_embeddings


class SDSLoss(nn.Module):
    def __init__(self, device, pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base",
                 guidance_scale=7.5, reduction='mean', loss_scale=1.0, t_min=0.02, t_max=0.98):
        super().__init__()

        self.guidance_scale = guidance_scale
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.text_encoder = SDSTextEncoder(device, pretrained_model_name_or_path)
        self.device = device

        # Use DDIM scheduler for faster sampling 
        self.noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

        self.unet = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            safety_checker=None,
        ).unet.to(self.device)
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",
        ).to(self.device)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.reduction = reduction
        self.loss_scale = loss_scale

        self.t_min = int(t_min * self.noise_scheduler.config.num_train_timesteps)
        self.t_max = int(t_max * self.noise_scheduler.config.num_train_timesteps) 

    def get_text_embeddings(self, prompt, negative_prompt=""):

        # text embeddings
        text_input = self.text_encoder.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.text_encoder.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder.text_encoder(text_input.input_ids.to(self.device))[0]

        # unconditional embeddings
        uncond_input = self.text_encoder.tokenizer(
            [negative_prompt] * (len(prompt)),  # Ensure correct batch size for uncond
            padding="max_length",
            max_length=self.text_encoder.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for CFG
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    def forward(self, images, text_embeddings, batch_size):
        r"""Compute the SDS loss.

        Args:
            images (torch.Tensor): Batch of input images (B, C, H, W), values in [-1, 1].
            text_embeddings (torch.Tensor): Text embeddings from the text encoder.
        Returns:
            torch.Tensor: SDS loss value.
        """
        with torch.amp.autocast('cuda', enabled=True):
            # Image to latent
            images = images * 0.5 + 0.5  # De-normalize from [-1, 1] to [0, 1]
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # Scale by the VAE scaling factor
            latents = latents.to(self.device)

            # Sample a timestep t.
            timesteps = torch.randint(self.t_min, self.t_max + 1, (batch_size,), device="cuda", dtype=torch.long)

            # Add noise to the images (forward diffusion process)
            noise = torch.randn_like(latents)
            noisy_images = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
            # Get the predicted noise 
            latent_model_input = torch.cat([noisy_images] * 2)
            noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings).sample

            # Classifier-free guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            #predict the 'foregound' and take the gradient on it as the SDS loss. (v-objective in ldm)
            w = (1 - self.noise_scheduler.alphas_cumprod[timesteps])
            grad = w[:, None, None, None] * (noise_pred - noise)

            grad = torch.nan_to_num(grad)
            target = (latents - grad)
            loss = 0.5 * F.mse_loss(latents, target, reduction='none') 
            loss = loss.mean()
            
        return loss * self.loss_scale