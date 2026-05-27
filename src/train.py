import os
import time
import numpy as np
from src.utils import load_dataset, normalize_vector


def build_data_matrix(vectors: np.ndarray) -> tuple:
    # Normalize each image to zero mean, unit variance before building the subspace.
    # This makes the SVD capture facial structure (edges, contours, symmetry)
    # rather than global illumination levels, which improves non-face rejection.
    normalized = []
    for i in range(vectors.shape[1]):
        v = normalize_vector(vectors[:, i])
        if v is not None:
            normalized.append(v)

    A = np.column_stack(normalized).astype(np.float32)
    mean_face = A.mean(axis=1)
    A_centered = A - mean_face[:, np.newaxis]
    return A_centered, mean_face.astype(np.float32)


def compute_svd(A: np.ndarray) -> tuple:
    print(f"Computing SVD on matrix of shape {A.shape} ...")
    t0 = time.time()
    U, s, Vt = np.linalg.svd(A.astype(np.float32), full_matrices=False)
    elapsed = time.time() - t0
    print(f"SVD complete in {elapsed:.1f}s")
    print(f"Top 5 singular values: {s[:5]}")
    return U, s, Vt


def select_k(singular_values: np.ndarray, variance_threshold: float = 0.95) -> int:
    total = np.sum(singular_values ** 2)
    cumulative = np.cumsum(singular_values ** 2) / total
    k = int(np.searchsorted(cumulative, variance_threshold)) + 1
    return k


def calibrate_threshold(A_centered: np.ndarray, U_k: np.ndarray,
                        percentile: float = 95.0) -> tuple:
    # Project all training faces onto the face subspace and measure reconstruction error
    coeffs = U_k.T @ A_centered      # (k, N)
    recon = U_k @ coeffs             # (10000, N)
    residuals = A_centered - recon   # (10000, N)
    errors = np.linalg.norm(residuals, axis=0)  # (N,)
    threshold = float(np.percentile(errors, percentile))
    return threshold, errors


def save_model(path: str, U_k: np.ndarray, mean_face: np.ndarray,
               threshold: float, singular_values: np.ndarray, k: int) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    np.savez_compressed(
        path,
        U_k=U_k.astype(np.float32),
        mean_face=mean_face.astype(np.float32),
        threshold=np.float32(threshold),
        singular_values=singular_values.astype(np.float32),
        k=np.int32(k),
    )
    print(f"Model saved → {path}")


def _save_plots(singular_values: np.ndarray, errors: np.ndarray,
                threshold: float, model_out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(model_out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Scree plot: singular value spectrum
    fig, ax = plt.subplots(figsize=(10, 4))
    n = min(200, len(singular_values))
    ax.plot(range(1, n + 1), singular_values[:n], "b-", linewidth=1.5)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular Value Spectrum (first 200 components)")
    ax.grid(True, alpha=0.3)
    scree_path = os.path.join(out_dir, "scree.png")
    fig.savefig(scree_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scree plot saved → {scree_path}")

    # Explained variance curve
    total = np.sum(singular_values ** 2)
    evr = np.cumsum(singular_values ** 2) / total
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(evr) + 1), evr * 100, "g-", linewidth=1.5)
    ax.axhline(95, color="red", linestyle="--", label="95% threshold")
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Explained Variance vs. k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    evr_path = os.path.join(out_dir, "explained_variance.png")
    fig.savefig(evr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Explained variance plot saved → {evr_path}")

    # Reconstruction error histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(threshold, color="red", linewidth=2,
               label=f"Threshold = {threshold:.1f}")
    ax.set_xlabel("Reconstruction error (L2 norm)")
    ax.set_ylabel("Count")
    ax.set_title("Training Face Reconstruction Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    hist_path = os.path.join(out_dir, "error_hist.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Error histogram saved → {hist_path}")


def train(data_dir: str, model_out: str, variance_threshold: float = 0.95,
          threshold_percentile: float = 95.0, plot: bool = False) -> None:
    print(f"Loading dataset from: {data_dir}")
    vectors = load_dataset(data_dir)
    N = vectors.shape[1]
    print(f"Loaded {N} images — matrix shape: {vectors.shape}")

    A_centered, mean_face = build_data_matrix(vectors)

    U, s, Vt = compute_svd(A_centered)

    k = select_k(s, variance_threshold)
    explained = float(np.sum(s[:k] ** 2) / np.sum(s ** 2)) * 100
    print(f"Selected k = {k} components ({explained:.2f}% variance explained)")

    U_k = U[:, :k].astype(np.float32)

    threshold, errors = calibrate_threshold(A_centered, U_k, threshold_percentile)
    print(f"Threshold (p{threshold_percentile:.0f}): {threshold:.2f}")
    print(f"Training errors — mean: {errors.mean():.2f}  std: {errors.std():.2f}")

    if plot:
        _save_plots(s, errors, threshold, model_out)

    save_model(model_out, U_k, mean_face, threshold, s, k)
