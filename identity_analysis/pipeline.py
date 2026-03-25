"""Pipeline wrappers for SDXL and Lumina2 with latent dumping callbacks."""

import os
import gc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*upcast_vae.*")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")


def _latent_dump_callback(latent_dir, step_latents: list):
    """Create a callback that captures latents at every denoising step."""
    def callback(pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        latent_np = latents.cpu().float().numpy()
        step_latents.append(latent_np.copy())
        if latent_dir is not None:
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
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        # Clear any stale VRAM before loading
        torch.cuda.empty_cache()
        gc.collect()

        # Use the fp16-fixed VAE to avoid numerical instability and float32 upcast
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.to("cuda")
        self.pipe.enable_vae_tiling()
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
            self.pipe.enable_model_cpu_offload()
            self.vae = self.pipe.vae
            self.vae_scale_factor = self.pipe.vae_scale_factor
            self.latent_channels = 16
        except Exception as e:
            print(f"Failed to load Lumina2: {e}")
            print("Falling back to SDXL.")
            self.model_type = "sdxl"
            self._load_sdxl()

    # Default negative prompt to steer SDXL away from anime/stylized outputs
    DEFAULT_NEGATIVE_PROMPT = (
        "anime, cartoon, illustration, painting, drawing, art, sketch, "
        "3d render, cgi, unrealistic, deformed, disfigured, bad anatomy, "
        "blurry, low quality, watermark, text"
    )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int = 20,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.5,
        output_dir: Optional[Path] = None,
        save_step_latents: bool = True,
        height: int = 1024,
        width: int = 1024,
    ) -> dict:
        """Generate an image and optionally dump latents at every step.

        Returns dict with keys: image, final_latent, step_latents (list of np arrays)
        """
        if negative_prompt is None:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT

        generator = torch.Generator(device="cpu").manual_seed(seed)

        step_latents = []
        latent_dir = None
        callback_fn = None

        if save_step_latents:
            if output_dir is not None:
                latent_dir = Path(output_dir) / "latents"
                latent_dir.mkdir(parents=True, exist_ok=True)
            callback_fn = _latent_dump_callback(latent_dir, step_latents)

        # Use a callback to capture the final latent before VAE decode
        final_latent_holder = {}

        def capture_final_latent(pipe, step_index, timestep, callback_kwargs):
            # Always capture the latest latent (last call = final step)
            final_latent_holder["latent"] = callback_kwargs["latents"].cpu().float().clone()
            if callback_fn is not None:
                callback_kwargs = callback_fn(pipe, step_index, timestep, callback_kwargs)
            return callback_kwargs

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            callback_on_step_end=capture_final_latent,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        result = self.pipe(**kwargs)
        image = result.images[0]
        final_latent_np = final_latent_holder["latent"].numpy()

        del result
        torch.cuda.empty_cache()

        return {
            "image": image,
            "final_latent": final_latent_np,
            "step_latents": step_latents,
        }

    @torch.no_grad()
    def generate_pair(
        self,
        prompt_a: str,
        prompt_b: str,
        seed: int,
        num_inference_steps: int = 20,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        decode_images: bool = False,
    ) -> tuple[dict, dict]:
        """Generate a pair of images in a single batched forward pass.

        ~2x faster than two separate generate() calls. Skips VAE decode
        by default since experiments only need latents.

        Returns (result_a, result_b) each with keys: final_latent, image (None if not decoded)
        """
        if negative_prompt is None:
            negative_prompt = self.DEFAULT_NEGATIVE_PROMPT

        generator_a = torch.Generator(device="cpu").manual_seed(seed)
        generator_b = torch.Generator(device="cpu").manual_seed(seed)

        # Capture latents from the batched output
        final_latents_holder = {}

        def capture_latents(pipe, step_index, timestep, callback_kwargs):
            final_latents_holder["latents"] = callback_kwargs["latents"].cpu().float().clone()
            return callback_kwargs

        # Batch both prompts together
        kwargs = dict(
            prompt=[prompt_a, prompt_b],
            negative_prompt=[negative_prompt, negative_prompt],
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=[generator_a, generator_b],
            output_type="pil" if decode_images else "latent",
            callback_on_step_end=capture_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        result = self.pipe(**kwargs)

        latents = final_latents_holder["latents"].numpy()
        lat_a = latents[0:1]  # [1, C, H, W]
        lat_b = latents[1:2]

        img_a = result.images[0] if decode_images else None
        img_b = result.images[1] if decode_images else None

        del result
        torch.cuda.empty_cache()

        return (
            {"image": img_a, "final_latent": lat_a, "step_latents": []},
            {"image": img_b, "final_latent": lat_b, "step_latents": []},
        )

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode an image through the VAE encoder to get a clean latent."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
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
        torch.cuda.empty_cache()
        gc.collect()
