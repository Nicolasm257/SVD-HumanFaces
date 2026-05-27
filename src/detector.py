import os
import numpy as np
from src.utils import load_image_as_vector, preprocess_frame, get_crop_rect, normalize_vector


def load_model(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            f"Run training first:\n"
            f"  python main.py train --data-dir \"data/Human Faces Dataset/Real Images\""
        )
    data = np.load(path)
    return {
        "U_k": data["U_k"],
        "mean_face": data["mean_face"],
        "threshold": float(data["threshold"]),
        "singular_values": data["singular_values"],
        "k": int(data["k"]),
    }


def compute_reconstruction_error(x: np.ndarray, U_k: np.ndarray,
                                  mean_face: np.ndarray) -> float:
    # Normalize per-image (must match training preprocessing)
    x_norm = normalize_vector(x)
    if x_norm is None:
        return float("inf")  # uniform image → not a face

    # Center using the mean of normalized training faces
    centered = x_norm - mean_face
    # Project onto the face subspace (coordinates in eigenface basis)
    coords = U_k.T @ centered
    # Reconstruct from the face subspace
    recon = U_k @ coords
    # Euclidean norm of the residual
    return float(np.linalg.norm(centered - recon))


def classify(error: float, threshold: float) -> str:
    return "Human Face" if error < threshold else "Not a Face"


def classify_image_file(image_path: str, model: dict) -> dict:
    x = load_image_as_vector(image_path)
    error = compute_reconstruction_error(x, model["U_k"], model["mean_face"])
    label = classify(error, model["threshold"])
    return {
        "path": image_path,
        "label": label,
        "error": round(error, 2),
        "threshold": round(model["threshold"], 2),
        "margin": round(model["threshold"] - error, 2),
    }


def run_camera_mode(model: dict, camera_id: int = 0) -> None:
    import cv2

    U_k = model["U_k"]
    mean_face = model["mean_face"]
    threshold = model["threshold"]

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera with index {camera_id}. "
            f"Try a different --camera-id value."
        )

    print("Camera active. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x = preprocess_frame(frame)
        error = compute_reconstruction_error(x, U_k, mean_face)
        label = classify(error, threshold)

        pt1, pt2 = get_crop_rect(frame.shape)
        is_face = label == "Human Face"
        color = (0, 200, 0) if is_face else (0, 0, 210)

        # Draw analysis rectangle and label
        cv2.rectangle(frame, pt1, pt2, color, 2)
        cv2.putText(
            frame, label,
            (pt1[0], pt1[1] - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
        )

        # Draw numeric info at the bottom
        h = frame.shape[0]
        cv2.putText(
            frame,
            f"Error: {error:.1f}  |  Threshold: {threshold:.1f}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
        )

        cv2.imshow("SVD Face Detector  [Q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
