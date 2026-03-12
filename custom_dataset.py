"""
custom_dataset.py

Utility for loading a custom CSV/JSON text-classification dataset
and converting it into a Hugging Face DatasetDict ready for fine-tuning.

Expected CSV/JSON schema:
  - "text"  : the input string
  - "label" : integer class id (0-indexed)

Usage:
    from custom_dataset import load_custom_dataset
    ds = load_custom_dataset("data/train.csv", "data/val.csv", test_size=0.1)
"""

from __future__ import annotations

import pathlib

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset


def load_custom_dataset(
    train_path: str,
    val_path: str | None = None,
    *,
    text_col: str = "text",
    label_col: str = "label",
    label_names: list[str] | None = None,
    test_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Load a CSV or JSON dataset and return a DatasetDict with
    'train' and 'validation' splits.

    Parameters
    ----------
    train_path : Path to the training file (.csv or .json).
    val_path   : Optional path to a separate validation file.
                 If None, ``test_size`` fraction of train is used.
    text_col   : Column name containing the raw text.
    label_col  : Column name containing integer labels.
    label_names: Human-readable label names in order.
                 If None, str(label_id) is used.
    test_size  : Fraction of train to hold out when val_path is None.
    seed       : Random seed for reproducibility.
    """
    ext = pathlib.Path(train_path).suffix.lower()
    fmt = "csv" if ext == ".csv" else "json"

    train_ds: Dataset = load_dataset(fmt, data_files={"train": train_path})["train"]

    if val_path:
        val_ds: Dataset = load_dataset(fmt, data_files={"validation": val_path})["validation"]
        splits = DatasetDict({"train": train_ds, "validation": val_ds})
    else:
        split = train_ds.train_test_split(test_size=test_size, seed=seed)
        splits = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Rename columns if needed
    if text_col != "text":
        splits = splits.rename_column(text_col, "text")
    if label_col != "label":
        splits = splits.rename_column(label_col, "label")

    # Cast label to ClassLabel for cleaner downstream handling
    if label_names:
        class_label = ClassLabel(names=label_names)
        features = Features({"text": Value("string"), "label": class_label})
        splits = splits.cast(features)

    return splits
