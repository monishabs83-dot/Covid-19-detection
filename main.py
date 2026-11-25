"""
PulmoNet AI - Entrypoint Script.

Usage:
  python main.py web        # Start the web application on http://127.0.0.1:8000
  python main.py train      # Train the CovidCNN model
  python main.py evaluate   # Run test evaluation and generate diagnostic plots
  python main.py generate   # Regenerate synthetic CXR dataset and sample images
"""

import sys
import argparse
import uvicorn
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="PulmoNet AI: COVID-19 CXR Detection Suite")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Web sub-command
    web_parser = subparsers.add_parser("web", help="Launch the interactive web application")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    web_parser.add_argument("--port", type=int, default=8001, help="Port (default: 8001)")
    web_parser.add_argument("--reload", action="store_true", help="Enable live auto-reload")

    # Train sub-command
    train_parser = subparsers.add_parser("train", help="Train the CovidCNN model")
    train_parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Evaluate sub-command
    subparsers.add_parser("evaluate", help="Evaluate the model on test split and generate plots")

    # Generate sub-command
    subparsers.add_parser("generate", help="Generate CXR dataset and curated demonstration cases")

    args = parser.parse_args()

    if args.command == "web" or args.command is None:
        print("\n" + "=" * 65)
        print("🫁 Launching PulmoNet AI Web Application...")
        print(f"👉 Open in browser: http://{getattr(args, 'host', '127.0.0.1')}:{getattr(args, 'port', 8001)}")
        print("=" * 65 + "\n")
        uvicorn.run("web.app:app", host=getattr(args, "host", "127.0.0.1"), port=getattr(args, "port", 8001), reload=getattr(args, "reload", False))

    elif args.command == "train":
        from src.train import train_model
        train_model(num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    elif args.command == "evaluate":
        from src.evaluate import compute_metrics_and_plots
        compute_metrics_and_plots()

    elif args.command == "generate":
        from src.data_generator import generate_full_dataset, generate_curated_samples
        generate_full_dataset()
        generate_curated_samples()


if __name__ == "__main__":
    main()
