from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def validate_aggregation_method(method: str) -> str:
    allowed = {"mean", "max"}
    if method not in allowed:
        raise ValueError(
            f"Invalid aggregation_method {method!r}. Choose one of {sorted(allowed)}."
        )
    return method


def build_model(args) -> Pipeline:
    steps = []
    if not args.no_scale:
        steps.append(("scaler", StandardScaler()))

    clf = LogisticRegression(
        max_iter=args.max_iter,
        class_weight="balanced",
        random_state=int(args.random_state),
    )

    steps.append(("clf", clf))
    return Pipeline(steps)
