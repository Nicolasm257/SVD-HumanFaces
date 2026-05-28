# SVD Human Face Detector

A human face classifier built on **Singular Value Decomposition (SVD)**. The model learns a low-dimensional eigenface subspace from a dataset of real face images and classifies new images by measuring their reconstruction error against that subspace. No neural networks — pure linear algebra.

---

## How it works

1. **Train** — load 5,000 grayscale face images (100×100), normalize each one, build a matrix A (10,000 × 5,000), and compute the thin SVD: `A = U Σ Vᵀ`
2. **Truncate** — keep the top k=512 columns of U (the *eigenfaces*) that explain 95% of the total variance
3. **Classify** — for any new image, project it onto the eigenface subspace, reconstruct it, and measure the residual error using the Euclidean norm
   - `error < threshold (27.79)` → **Human Face**
   - `error ≥ threshold` → **Not a Face**

---

## Project structure

```
SVD-HumanFaces/
├── src/
│   ├── utils.py       # image loading, preprocessing, per-image normalization
│   ├── train.py       # SVD training pipeline, threshold calibration, model export
│   └── detector.py    # reconstruction error classifier, webcam loop
├── models/
│   ├── face_model.npz          # pre-trained SVD model (k=512, θ=27.79)
│   ├── scree.png
│   ├── explained_variance.png
│   └── error_hist.png
├── main.py            # CLI entry point (train / detect --image / detect --camera)
└── requirements.txt
```

> **Note:** `data/` (training images) is excluded from this repository. A pre-trained model is included in `models/` — training is optional.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** `numpy`, `Pillow`, `opencv-python`, `tqdm`, `matplotlib`, `scipy`

---

## Dataset setup

Download the [Human Faces Dataset](https://www.kaggle.com/datasets/ashwingupta3012/human-faces) and place it as:

```
data/
└── Human Faces Dataset/
    ├── Real Images/          # 5,000 JPGs  (178×218)
    └── AI-Generated Images/  # 4,630 JPGs  (256×256)
```

---

## Usage

### 1. Train the model *(optional)*

A pre-trained model is included in `models/face_model.npz`. Training is only needed if you want to retrain from scratch.

```bash
python main.py train --data-dir "data/Human Faces Dataset/Real Images"
```

Optional flags:
```
--model-out models/face_model.npz   output path (default: models/face_model.npz)
--variance-threshold 0.95           fraction of variance to retain for k selection
--threshold-percentile 95           percentile of training errors used as threshold
--plot                              save scree.png, explained_variance.png, error_hist.png
```

Training takes ~2 minutes on Apple Silicon. The resulting model (`face_model.npz`) stores the eigenface matrix U_k, the mean face vector, the threshold θ, and the singular value spectrum.

---

### 2. Classify an image

```bash
python main.py detect --image path/to/image.jpg
```

Example output:
```
  Image:      path/to/image.jpg
  Result:     Human Face
  Error:      18.43  (threshold: 27.79)
  Confidence: 34% below threshold
```

---

### 3. Live webcam detection

```bash
python main.py detect --camera
```

An OpenCV window opens showing the camera feed. A centered rectangle marks the region being analyzed — position your face inside it. The label and reconstruction error update in real time. Press **Q** to quit.

```
--camera-id 0    webcam device index (default: 0)
```

---

## Model details

| Parameter | Value |
|---|---|
| Training images | 5,000 real human faces |
| Image size | 100 × 100 px (grayscale) |
| Normalization | per-image zero mean, unit variance |
| SVD components (k) | 512 |
| Variance explained | 95.01% |
| Decision threshold θ | 27.79 |
| Accuracy on held-out faces | 95 / 100 |

---

## Diagnostic plots

Running `--plot` during training saves three figures to the model directory:

| File | What it shows |
|---|---|
| `scree.png` | Singular value spectrum — confirms information is concentrated in early components |
| `explained_variance.png` | Cumulative variance vs. k — shows where 95% threshold is crossed |
| `error_hist.png` | Distribution of training reconstruction errors with threshold line |

---

## Known limitations

The classifier operates on a global 100×100 pixel representation. Any image whose low-resolution normalized pixel distribution aligns with the eigenface subspace will be accepted as a face, regardless of semantic content. Objects with a bright center and dark periphery (e.g. a metallic cube under direct lighting) can produce false positives. The model does not use local feature extraction; it is a statistical similarity test against the training distribution, not a semantic face detector.

---

## Academic context

This project was developed for an Applied Mathematics course to demonstrate the connection between Singular Value Decomposition and practical computer vision.
