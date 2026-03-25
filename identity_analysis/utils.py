"""General utilities: face detection, latent-space mapping, output management."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def get_output_dir(experiment_name: str, base_dir: str = "outputs") -> Path:
    """Create and return a timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / timestamp / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    (out / "latents").mkdir(exist_ok=True)
    return out


def pixel_bbox_to_latent_bbox(
    pixel_bbox: tuple[float, float, float, float],
    vae_scale_factor: int = 8,
    image_size: tuple[int, int] = (512, 512),
    latent_size: Optional[tuple[int, int]] = None,
) -> tuple[int, int, int, int]:
    """Map a pixel-space bounding box to latent-space coordinates.

    Args:
        pixel_bbox: (x1, y1, x2, y2) in pixel space
        vae_scale_factor: Downscale factor of the VAE (typically 8)
        image_size: (width, height) of the image
        latent_size: (height, width) of the latent, computed if None

    Returns:
        (y1, x1, y2, x2) in latent space (note: y,x order for array indexing)
    """
    x1, y1, x2, y2 = pixel_bbox

    if latent_size is None:
        lh = image_size[1] // vae_scale_factor
        lw = image_size[0] // vae_scale_factor
    else:
        lh, lw = latent_size

    scale_x = lw / image_size[0]
    scale_y = lh / image_size[1]

    lx1 = max(0, int(x1 * scale_x))
    ly1 = max(0, int(y1 * scale_y))
    lx2 = min(lw, int(np.ceil(x2 * scale_x)))
    ly2 = min(lh, int(np.ceil(y2 * scale_y)))

    return (ly1, lx1, ly2, lx2)


def detect_face_bbox(image: Image.Image) -> Optional[tuple[float, float, float, float]]:
    """Detect the largest face in an image and return its pixel bbox.

    Returns (x1, y1, x2, y2) or None if no face found.
    """
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))

        img_np = np.array(image.convert("RGB"))[:, :, ::-1]
        faces = app.get(img_np)

        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return tuple(face.bbox.tolist())
    except Exception:
        return None


def crop_face(image: Image.Image, padding: float = 0.2) -> Optional[Image.Image]:
    """Detect and crop the face from an image with padding."""
    bbox = detect_face_bbox(image)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * padding)
    y1 = max(0, y1 - h * padding)
    x2 = min(image.width, x2 + w * padding)
    y2 = min(image.height, y2 + h * padding)

    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def get_prompt_pairs() -> list[tuple[str, str]]:
    """Return prompt pairs for paired latent analysis.

    Each pair has the same scene but different identity subjects.
    """
    return [
        # Same gender, different identity (isolate identity from gender)
        (
            "portrait photo of a young man in a cafe, natural lighting",
            "portrait photo of an old man in a cafe, natural lighting",
        ),
        (
            "portrait photo of a young woman in a park, natural lighting",
            "portrait photo of an old woman in a park, natural lighting",
        ),
        # Cross-gender, same scene
        (
            "close-up portrait of a man in a studio, soft lighting",
            "close-up portrait of a woman in a studio, soft lighting",
        ),
        # Same identity description, different scene (control: identity constant, scene varies)
        (
            "photo of a middle-aged man at a desk, office background",
            "photo of a middle-aged man at the beach, sunset lighting",
        ),
        # Different ethnicity cues, same framing
        (
            "headshot of an Asian man, plain white background",
            "headshot of a European man, plain white background",
        ),
        (
            "headshot of an African woman, plain white background",
            "headshot of a European woman, plain white background",
        ),
        # Different ages, same gender, varied settings
        (
            "photo of a teenage boy standing on a street, urban, golden hour",
            "photo of an elderly man standing on a street, urban, golden hour",
        ),
        (
            "photo of a young woman sitting in a library, warm lighting",
            "photo of an elderly woman sitting in a library, warm lighting",
        ),
        # Full body vs portrait (composition variety)
        (
            "full body photo of a tall man in a suit, city sidewalk",
            "full body photo of a short man in a suit, city sidewalk",
        ),
        # Minimal prompt, maximum identity difference
        (
            "photo of a man",
            "photo of a woman",
        ),
        (
            "photo of a child",
            "photo of an old person",
        ),
        # Same scene, subtle identity shift
        (
            "portrait of a man with short brown hair, neutral expression, grey background",
            "portrait of a man with long blond hair, neutral expression, grey background",
        ),
    ]


def get_celebrity_prompts() -> dict[str, list[str]]:
    """Return varied prompts for each celebrity for within-identity analysis."""
    celebrities = [
        "Brad Pitt",
        "Taylor Swift",
        "Morgan Freeman",
        "Scarlett Johansson",
        "Keanu Reeves",
    ]

    templates = [
        "portrait photo of {name}, studio lighting, neutral background",
        "photo of {name} outdoors in a park, natural lighting",
        "close-up of {name}, dramatic side lighting",
        "photo of {name} in a suit at a formal event",
        "candid photo of {name} in casual clothes, street photography",
        "photo of {name} smiling, bright lighting, white background",
        "black and white portrait of {name}",
        "photo of {name} in a restaurant, warm ambient lighting",
        "headshot of {name}, professional photography",
        "photo of {name} at the beach, sunset lighting",
        "photo of {name} reading a book in a library",
        "photo of {name} wearing a hat, outdoor photography",
        "close-up portrait of {name}, bokeh background",
        "photo of {name} standing near a window, natural light",
        "cinematic still of {name}, film grain, moody lighting",
        "photo of {name} in a garden, soft morning light",
        "photo of {name} in winter clothes, snowy background",
        "photo of {name} leaning against a wall, urban setting",
        "editorial portrait of {name}, magazine quality",
        "photo of {name} sitting in a cafe, cozy atmosphere",
    ]

    prompts = {}
    for celeb in celebrities:
        prompts[celeb] = [t.format(name=celeb) for t in templates]

    return prompts
