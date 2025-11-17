# src/data_handler.py
import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from scripts.config import DataConfig

COLUMN_NAMES = [
    "ip_mean",
    "ip_std",
    "ip_kurtosis",
    "ip_skewness",
    "dm_mean",
    "dm_std",
    "dm_kurtosis",
    "dm_skewness",
    "signal",
]


class HTRU2DataHandler:
    """Handles downloading, loading, and preprocessing of the HTRU2 dataset."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.data_file = os.path.join(self.data_dir, config.external_dir, "HTRU_2.csv")
        self.output_file = os.path.join(
            self.data_dir, config.processed_dir, "HTRU2_cleaned_data.csv"
        )
        self.df = None
        self.splits = {}

        os.makedirs(self.data_dir, exist_ok=True)

    def download_kaggle(self):
        """Download dataset from Kaggle if not found locally."""
        if os.path.exists(self.data_file):
            print("✅ Dataset already exists locally.")
            return

        print("⬇️  Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.config.dataset, path=self.data_dir, unzip=True)

        # Unzip and cleanup & rename the file if necessary
        for f in os.listdir(self.data_dir):
            if f.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(self.data_dir, f), "r") as zip_ref:
                    zip_ref.extractall(self.data_dir)
                os.remove(os.path.join(self.data_dir, f))

        if not os.path.exists(self.data_file):
            # Assuming the extracted file has a different name
            extracted_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
            if extracted_files:
                os.rename(
                    os.path.join(self.data_dir, extracted_files[0]),
                    self.data_file,
                )

        print("✅ Download complete.")

    def load(self):
        """Load the dataset into a pandas DataFrame."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"{self.data_file} not found. Run download() first.")
        print("📂 Loading data...")
        self.df = pd.read_csv(self.data_file)
        print(f"✅ Loaded {len(self.df)} rows.")
        return self.df

    def get_column_types(self):
        """Get the data types of each column in the DataFrame."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        # Check numerical values
        features = self.df.columns[(self.df.columns != "signal")].tolist()
        numerical_cols = (
            self.df[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
        )
        categorical_cols = (
            self.df[features].select_dtypes(include=["object", "category"]).columns.tolist()
        )
        return numerical_cols, categorical_cols

    def preprocess(self):
        """Preprocess the dataset: rename columns, check for missing values, and round numerical values."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        print("⚙️  Preprocessing data...")

        # Rename columns
        self.df.columns = COLUMN_NAMES

        # Check for missing values
        if self.df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains missing values.")

        numerical_cols, _ = self.get_column_types()

        # Apply rounding
        self.df[numerical_cols] = self.df[numerical_cols].apply(lambda x: round(x, 3))

        print("✅ Processing complete.")
        return self.df

    def split_train_val_test(self):
        """Split the dataset into training, validation, and test sets."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        print("✂️  Splitting data into train, val, and test sets...")

        full_train_df, test_df = train_test_split(
            self.df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=self.df["signal"],
        )

        train_df, val_df = train_test_split(
            self.df,
            test_size=self.config.val_size,
            random_state=self.config.random_state,
            stratify=self.df["signal"],
        )

        self.splits = {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

        print("✅ Data splitting complete.")
        return self.splits

    def split_fulltrain_test(self):
        """Split the dataset into full training and test sets."""

        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        print("✂️  Splitting data into train, val, and test sets...")

        full_train_df, test_df = train_test_split(
            self.df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=self.df["signal"],
        )

        self.splits = {
            "train": full_train_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

        print("✅ Data splitting complete.")
        return self.splits

    def export_splits(self):
        """Export the train, validation, and test splits to CSV files."""
        if not self.splits:
            raise ValueError("Data splits not found. Call split_train_val_test() first.")

        print("💾 Exporting data splits to CSV files...")
        for split_name, df_split in self.splits.items():
            file_path = os.path.join(self.output_file, f"{split_name}_data.csv")
            df_split.to_csv(file_path, index=False)
            print(f"✅ Exported {split_name} set to {file_path}.")

    def export_cleaned_data(self):
        """Export the cleaned full dataset to a CSV file."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        file_path = os.path.join(self.output_file, "HTR2_cleaned_data.csv")
        self.df.to_csv(file_path, index=False)
        print(f"✅ Exported cleaned data to {file_path}.")
