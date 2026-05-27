import argparse
import sys


def cmd_train(args):
    from src.train import train
    train(
        data_dir=args.data_dir,
        model_out=args.model_out,
        variance_threshold=args.variance_threshold,
        threshold_percentile=args.threshold_percentile,
        plot=args.plot,
    )


def cmd_detect(args):
    from src.detector import load_model, classify_image_file, run_camera_mode

    model = load_model(args.model)

    if args.camera:
        run_camera_mode(model, camera_id=args.camera_id)
    elif args.image:
        result = classify_image_file(args.image, model)
        t = result["threshold"]
        e = result["error"]
        # Confidence: how far the error is from the threshold, as a percentage of threshold
        confidence = abs(result["margin"]) / t * 100
        confidence_str = f"{confidence:.0f}% {'above' if e > t else 'below'} threshold"
        print()
        print(f"  Image:      {result['path']}")
        print(f"  Result:     {result['label']}")
        print(f"  Error:      {e}  (threshold: {t})")
        print(f"  Confidence: {confidence_str}")
        print()
    else:
        print("Error: provide --image <path> or --camera", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SVD Face Detector\n"
            "Classifies images as 'Human Face' or 'Not a Face' using\n"
            "Singular Value Decomposition and reconstruction error."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    tp = subparsers.add_parser(
        "train",
        help="Build the SVD face model from a folder of training images",
    )
    tp.add_argument(
        "--data-dir", required=True,
        help='Path to folder containing face training images (e.g. "data/Human Faces Dataset/Real Images")',
    )
    tp.add_argument(
        "--model-out", default="models/face_model.npz",
        help="Output path for the saved model (default: models/face_model.npz)",
    )
    tp.add_argument(
        "--variance-threshold", type=float, default=0.95,
        help="Fraction of variance retained when selecting k (default: 0.95)",
    )
    tp.add_argument(
        "--threshold-percentile", type=float, default=95.0,
        help="Percentile of training errors used as decision threshold (default: 95)",
    )
    tp.add_argument(
        "--plot", action="store_true",
        help="Save diagnostic plots (scree.png, error_hist.png) alongside the model",
    )

    # ── detect ─────────────────────────────────────────────────────────────
    dp = subparsers.add_parser(
        "detect",
        help="Classify a single image or run live camera detection",
    )
    dp.add_argument(
        "--model", default="models/face_model.npz",
        help="Path to a trained model file (default: models/face_model.npz)",
    )
    mode = dp.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--image", metavar="PATH",
        help="Path to an image file to classify",
    )
    mode.add_argument(
        "--camera", action="store_true",
        help="Open webcam for real-time face detection",
    )
    dp.add_argument(
        "--camera-id", type=int, default=0,
        help="Webcam device index (default: 0)",
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "detect":
        cmd_detect(args)


if __name__ == "__main__":
    main()
