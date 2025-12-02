# scripts/config.py
from pydantic_settings import BaseSettings
import os


class DataConfig(BaseSettings):
    # Use absolute paths based on project root
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(project_root, "data")
    external_dir: str = "external"
    processed_dir: str = "processed"
    dataset: str = "charitarth/pulsar-dataset-htru2"
    test_size: float = 0.2
    val_size: float = 0.25
    random_state: int = 42

    class Config:
        env_file = ".env"
