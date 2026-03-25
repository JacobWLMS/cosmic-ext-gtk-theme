"""Pipeline wrappers for SDXL and Lumina2 with latent dumping callbacks."""

import os
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _latent_dump_callback(latent_dir: Path, step_latents: list):
    """Create a callback that dumps latents at every denoising step."""
    def callback(pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        latent_np = latents.cpu().float().numpy()
        step_latents.append(latent_np.copy())
        np.save(latent_dir / f"step_{step_index:04d}.npy", latent_np)
        return callback_kwargs
    return callback


class PipelineWrapper:
    """Unified pipeline wrapper for SDXL and Lumina2."""

    def __init__(self, model_type: str = "sdxl", device: str = "cuda"):
        self.model_type = model_type
        self.device = device
        self.pipe = None
        self.vae = None
        self._load_pipeline()

    def _load_pipeline(self):
        if self.model_type == "sdxl":
            self._load_sdxl()
        elif self.model_type == "lumina2":
            self._load_lumina2()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_sdxl(self):
        from diffusers import StableDiffusionXLPipeline

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        if torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
        else:
            self.pipe = self.pipe.to("cpu")
            self.pipe.to(torch.float32)
        self.vae = self.pipe.vae
        self.vae_scale_factor = self.pipe.vae_scale_factor  # typically 8
        self.latent_channels = 4

    def _load_lumina2(self):
        try:
            from diffusers import Lumina2Pipeline

            self.pipe = Lumina2Pipeline.from_pretrained(
                "Alpha-VLLM/Lumina-Image-2.0",
                torch_dtype=torch.bfloat16,
            )
            if torch.cuda.is_available():
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe = self.pipe.to("cpu")
                self.pipe.to(torch.float32)
            self.vae = self.pipe.vae
            self.vae_scale_factor = self.pipe.vae_scale_factor
            self.latent_channels = 16
        except Exception as e:
            print(f"Failed to load Lumina2: {e}")
            print("Falling back to SDXL.")
            self.model_type = "sdxl"
            self._load_sdxl()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int = 50,
        output_dir: Optional[Path] = None,
        save_step_latents: bool = True,
        height: int = 512,
        width: int = 512,
    ) -> dict:
        """Generate an image and optionally dump latents at every step.

        Returns dict with keys: image, final_latent, step_latents (list of np arrays)
        """
        generator = torch.Generator(device="cpu").manual_seed(seed)

        step_latents = []
        latent_dir = None
        callback_fn = None

        if save_step_latents and output_dir is not None:
            latent_dir = Path(output_dir) / "latents"
            latent_dir.mkdir(parents=True, exist_ok=True)
            callback_fn = _latent_dump_callback(latent_dir, step_latents)

        kwargs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="latent",
        )

        if callback_fn is not None:
            kwargs["callback_on_step_end"] = callback_fn
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        result = self.pipe(**kwargs)
        final_latent = result.images  # in latent output mode, this is the latent

        # Decode the final latent to get the image
        if self.model_type == "sdxl":
            decoded = self.vae.decode(
                final_latent / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]
        else:
            decoded = self.vae.decode(
                final_latent / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]

        image = self.pipe.image_processor.postprocess(decoded, output_type="pil")[0]

        final_latent_np = final_latent.cpu().float().numpy()

        # Clean up
        del result, decoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "image": image,
            "final_latent": final_latent_np,
            "step_latents": step_latents,
        }

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode an image through the VAE encoder to get a clean latent."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        img_tensor = transform(image.convert("RGB")).unsqueeze(0)
        img_tensor = img_tensor.to(
            device=self.vae.device,
            dtype=self.vae.dtype,
        )

        latent_dist = self.vae.encode(img_tensor).latent_dist
        latent = latent_dist.sample() * self.vae.config.scaling_factor

        result = latent.cpu().float().numpy()
        del img_tensor, latent_dist, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    @torch.no_grad()
    def decode_latent(self, latent_np: np.ndarray) -> Image.Image:
        """Decode a latent tensor (numpy) through the VAE decoder to get an image."""
        latent = torch.from_numpy(latent_np).to(
            device=self.vae.device,
            dtype=self.vae.dtype,
        )
        decoded = self.vae.decode(
            latent / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        image = self.pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        del latent, decoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return image

    @torch.no_grad()
    def add_noise_to_latent(
        self, clean_latent_np: np.ndarray, timestep: int, seed: int
    ) -> np.ndarray:
        """Add noise to a clean latent to match a specific timestep (forward diffusion).

        Args:
            clean_latent_np: Clean latent as numpy array [1, C, H, W]
            timestep: The timestep to noise to (0 = clean, max = pure noise)
            seed: Random seed for noise generation

        Returns:
            Noised latent as numpy array
        """
        scheduler = self.pipe.scheduler
        clean_latent = torch.from_numpy(clean_latent_np).to(
            device="cpu", dtype=torch.float32
        )

        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(clean_latent.shape, generator=generator)

        timestep_tensor = torch.tensor([timestep], dtype=torch.long)
        noised = scheduler.add_noise(clean_latent, noise, timestep_tensor)

        return noised.numpy()

    def cleanup(self):
        """Free GPU memory."""
        del self.pipe
        del self.vae
        self.pipe = None
        self.vae = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
