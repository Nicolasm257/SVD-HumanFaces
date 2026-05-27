import os
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

IMAGE_SIZE = (100, 100)
VECTOR_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]


def load_image_as_vector(path: str, size: tuple = IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float32).flatten()


def load_dataset(folder: str, max_images: int = None) -> np.ndarray:
    extensions = (".jpg", ".jpeg", ".png")
    paths = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ])

    if max_images is not None:
        paths = paths[:max_images]

    vectors = []
    iterator = tqdm(paths, desc="Loading images") if _HAS_TQDM else paths

    for path in iterator:
        try:
            vectors.append(load_image_as_vector(path))
        except Exception:
            pass

    if not vectors:
        raise ValueError(f"No images found in: {folder}")

    return np.column_stack(vectors).astype(np.float32)


def preprocess_frame(bgr_frame: np.ndarray) -> np.ndarray:
    import cv2
    h, w = bgr_frame.shape[:2]
    sq = min(h, w)
    y1 = (h - sq) // 2
    x1 = (w - sq) // 2
    crop = bgr_frame[y1:y1 + sq, x1:x1 + sq]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return resized.astype(np.float32).flatten()


def get_crop_rect(frame_shape: tuple) -> tuple:
    h, w = frame_shape[:2]
    sq = min(h, w)
    y1 = (h - sq) // 2
    x1 = (w - sq) // 2
    return (x1, y1), (x1 + sq, y1 + sq)


def normalize_vector(x: np.ndarray) -> np.ndarray:
    """Per-image normalization: zero mean, unit variance.

    Removes global illumination so the face subspace captures structure,
    not absolute pixel values. Returns None for uniform images (std ≈ 0).
    """
    sigma = float(x.std())
    if sigma < 1.0:
        return None
    return ((x - x.mean()) / sigma).astype(np.float32)
