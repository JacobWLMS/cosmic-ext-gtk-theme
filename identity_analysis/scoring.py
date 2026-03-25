"""Scoring utilities: ArcFace, CLIP similarity, MSE, SSIM."""

import numpy as np
from PIL import Image


class ArcFaceScorer:
    """ArcFace cosine similarity scorer using insightface."""

    def __init__(self):
        self.app = None
        self._init_model()

    def _init_model(self):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image: Image.Image) -> np.ndarray | None:
        """Extract ArcFace embedding from a PIL image. Returns None if no face found."""
        img_np = np.array(image.convert("RGB"))
        # insightface expects BGR
        img_bgr = img_np[:, :, ::-1]
        faces = self.app.get(img_bgr)
        if len(faces) == 0:
            return None
        # Use the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding

    def similarity(self, img1: Image.Image, img2: Image.Image) -> float | None:
        """Compute ArcFace cosine similarity between two images.

        Returns None if face detection fails on either image.
        """
        emb1 = self.get_embedding(img1)
        emb2 = self.get_embedding(img2)
        if emb1 is None or emb2 is None:
            return None
        return float(np.dot(emb1, emb2))

    def detect_faces(self, image: Image.Image) -> list[dict]:
        """Detect faces and return bounding boxes and embeddings."""
        img_np = np.array(image.convert("RGB"))
        img_bgr = img_np[:, :, ::-1]
        faces = self.app.get(img_bgr)
        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.tolist(),  # [x1, y1, x2, y2]
                "embedding": face.normed_embedding,
                "det_score": float(face.det_score),
            })
        return results


class CLIPScorer:
    """CLIP similarity scorer using open_clip."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        import open_clip
        import torch

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def image_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute CLIP cosine similarity between two images."""
        import torch

        with torch.no_grad():
            feat1 = self.model.encode_image(
                self.preprocess(img1).unsqueeze(0).to(self.device)
            )
            feat2 = self.model.encode_image(
                self.preprocess(img2).unsqueeze(0).to(self.device)
            )
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
            return float((feat1 @ feat2.T).item())


def compute_mse(img1: Image.Image, img2: Image.Image) -> float:
    """Compute MSE between two PIL images."""
    arr1 = np.array(img1.convert("RGB")).astype(np.float32) / 255.0
    arr2 = np.array(img2.convert("RGB")).astype(np.float32) / 255.0
    if arr1.shape != arr2.shape:
        arr2 = np.array(img2.resize(img1.size).convert("RGB")).astype(np.float32) / 255.0
    return float(np.mean((arr1 - arr2) ** 2))


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Compute SSIM between two PIL images."""
    from skimage.metrics import structural_similarity

    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))
    if arr1.shape != arr2.shape:
        arr2 = np.array(img2.resize(img1.size).convert("RGB"))
    return float(structural_similarity(arr1, arr2, channel_axis=2))
