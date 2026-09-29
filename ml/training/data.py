"""Load generated transactions, compute features, split by time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.io import iter_transactions, read_parquet
from ml.features import FEATURE_NAMES
from ml.features.batch import featurize
from ml.training.config import SplitConfig


@dataclass(frozen=True)
class Dataset:
    features: pd.DataFrame  # columns == FEATURE_NAMES, float64
    label: np.ndarray  # 1 = fraud
    pattern: list[str | None]
    timestamp: pd.Series

    def __len__(self) -> int:
        return len(self.label)

    def take(self, mask: np.ndarray) -> Dataset:
        idx = np.flatnonzero(mask)
        return Dataset(
            features=self.features.iloc[idx].reset_index(drop=True),
            label=self.label[idx],
            pattern=[self.pattern[i] for i in idx],
            timestamp=self.timestamp.iloc[idx].reset_index(drop=True),
        )

    def to_frame(self) -> pd.DataFrame:
        """Features plus label columns: the logged test-set artifact."""
        return self.features.assign(
            is_fraud=self.label,
            fraud_pattern=pd.Series(self.pattern, dtype="object"),
            timestamp=self.timestamp,
        )

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> Dataset:
        return cls(
            features=df[list(FEATURE_NAMES)].astype("float64"),
            label=df["is_fraud"].to_numpy(dtype=np.int64),
            pattern=[p if isinstance(p, str) else None for p in df["fraud_pattern"]],
            timestamp=df["timestamp"],
        )


@dataclass(frozen=True)
class Split:
    train: Dataset
    valid: Dataset
    test: Dataset
    valid_start: datetime
    test_start: datetime


def load_dataset(path: Path) -> tuple[Dataset, dict[str, str]]:
    """Validate every row against the schema and featurise with the batch path.

    Returns the dataset and the generator metadata stored in the Parquet footer.
    Raises pydantic.ValidationError on schema violations.
    """
    table = read_parquet(path)
    rows = list(iter_transactions(table))
    features = pd.DataFrame(featurize(rows), columns=list(FEATURE_NAMES), dtype="float64")
    dataset = Dataset(
        features=features,
        label=np.array([r.is_fraud for r in rows], dtype=np.int64),
        pattern=[r.fraud_pattern for r in rows],
        timestamp=pd.Series([r.timestamp for r in rows]),
    )
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    return dataset, meta


def time_split(dataset: Dataset, config: SplitConfig) -> Split:
    """Oldest rows train; the next slice validates; the most recent slice tests.

    Features were computed on the whole stream first, so early test rows still
    see card history from before the boundary, exactly as they would online.
    """
    ts = dataset.timestamp
    start, end = ts.min(), ts.max()
    span = end - start
    valid_start = start + span * (1 - config.valid_fraction - config.test_fraction)
    test_start = start + span * (1 - config.test_fraction)
    in_train = (ts < valid_start).to_numpy()
    in_test = (ts >= test_start).to_numpy()
    in_valid = ~in_train & ~in_test
    return Split(
        train=dataset.take(in_train),
        valid=dataset.take(in_valid),
        test=dataset.take(in_test),
        valid_start=valid_start.to_pydatetime(),
        test_start=test_start.to_pydatetime(),
    )
