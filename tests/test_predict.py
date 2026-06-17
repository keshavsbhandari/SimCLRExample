import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from simclr_fraud.predict import score_dataframe


class _DummyClassifier(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def test_score_dataframe(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "name": "testexp",
            "data": {
                "drop_cols": ["isFraud", "isFlaggedFraud"],
                "target_col": "isFraud",
            },
        }
    )

    df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "amount": [10.0, 20.0, 30.0],
            "isFraud": [0, 1, 0],
            "isFlaggedFraud": [0, 0, 0],
        }
    )

    preproc = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["step", "amount"])],
        remainder="drop",
    )
    preproc.fit(df.drop(columns=["isFraud", "isFlaggedFraud"]))

    preproc_path = tmp_path / "preprocessor.joblib"
    import joblib

    joblib.dump(preproc, preproc_path)

    model = _DummyClassifier(input_dim=2)
    ckpt_path = tmp_path / "classifier.pt"
    torch.save(model.state_dict(), ckpt_path)

    monkeypatch.setattr("simclr_fraud.predict.preprocessor_path", lambda name: preproc_path)
    monkeypatch.setattr(
        "simclr_fraud.predict.load_classifier",
        lambda c, dim: model,
    )

    probs = score_dataframe(cfg, df, device=torch.device("cpu"))
    assert probs.shape == (3,)
    assert np.all((probs >= 0) & (probs <= 1))
