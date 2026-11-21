"""Walk-forward splits with purge gaps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass
class Fold:
    name: str
    train_dates: list[Any]
    tune_dates: list[Any] = field(default_factory=list)
    test_dates: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.train_dates = list(self.train_dates)
        self.tune_dates = list(self.tune_dates)
        self.test_dates = list(self.test_dates)
        if not self.train_dates:
            raise ValueError(f"Fold {self.name} has an empty train set")
        if not self.test_dates:
            raise ValueError(f"Fold {self.name} has an empty test set")


def rolling_window_folds(
    all_dates: Sequence[Any],
    test_dates_by_name: dict[Any, Sequence[Any]],
    train_window: int = 800,
    gap: int = 1,
    tune_size: int = 0,
) -> list[Fold]:
    """Build rolling-window folds.

    For each test block, the training window is the `train_window` dates
    immediately before `test_start - gap`. If `tune_size > 0`, the last
    `tune_size` training dates are moved into the tune segment.
    """
    ordered = list(all_dates)
    index = {value: i for i, value in enumerate(ordered)}
    folds: list[Fold] = []

    for name, raw_test in test_dates_by_name.items():
        test = list(raw_test)
        if not test:
            continue
        missing = [d for d in test if d not in index]
        if missing:
            raise ValueError(f"Fold {name}: {len(missing)} test dates are not in the panel")
        first_idx = index[test[0]]
        train_end = first_idx - int(gap)
        train_start = max(0, train_end - int(train_window))
        train = ordered[train_start:train_end]
        if not train:
            continue
        tune: list[Any] = []
        if tune_size > 0 and len(train) > tune_size:
            train, tune = train[:-tune_size], train[-tune_size:]
        folds.append(Fold(name=str(name), train_dates=train, tune_dates=tune, test_dates=test))

    if not folds:
        raise ValueError("No valid folds were produced")
    return folds
