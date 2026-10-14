"""Small EDA entry point for coursework exploration."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "data" / "heart_disease_cleaned.csv")
print("Shape:", data.shape); print("\nMissing values:\n", data.isna().sum()); print("\nClass distribution:\n", data["target"].value_counts().sort_index()); print("\nSummary statistics:\n", data.describe().T); print("\nRun `py train_model.py` from the project root to generate the EDA images.")
