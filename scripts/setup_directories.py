# src/setup_directories.py
import os


def setup_environment():
    """Create necessary directories for the project."""
    directories = [
        "data/external",
        "data/processed",
        "outputs/models",
        "outputs/metrics",
        "outputs/screenshots",
        "outputs/predictions",
        "logs",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Check if TOML config exists
    if not os.path.exists("model_config.toml"):
        print("❌ model_config.toml not found in current directory!")
        print("Please ensure your model_config.toml file is in the same directory as this script.")
    else:
        print("✅ model_config.toml found!")


if __name__ == "__main__":
    setup_environment()
