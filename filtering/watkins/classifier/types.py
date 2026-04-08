from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitData:
    name: str
    indices: np.ndarray
    manifest: pd.DataFrame
    X: np.ndarray
    y_text: np.ndarray
    y: np.ndarray
