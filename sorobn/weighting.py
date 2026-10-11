"""Row weights: fit from counts instead of repeated rows."""

import numpy as np
import pandas as pd

__all__ = ["split_weights"]


def split_weights(X, weights):
    """Return X without the weights column, and the weights as a float Series.

    ``weights`` is None (each row counts once), the name of a column of X, or
    one non-negative number per row. A row with weight 3 counts as 3 rows.
    """
    if weights is None:
        return X, pd.Series(1.0, index=X.index)

    if isinstance(weights, str):
        if weights not in X.columns:
            raise ValueError(f"weights column {weights!r} is not in the data")
        X, weights = X.drop(columns=weights), X[weights]

    values = np.asarray(weights, dtype=float)
    if values.shape != (len(X),):
        raise ValueError("weights must have one value per row")
    if not (np.isfinite(values) & (values >= 0)).all():
        raise ValueError("weights must be finite and non-negative")
    return X, pd.Series(values, index=X.index)
