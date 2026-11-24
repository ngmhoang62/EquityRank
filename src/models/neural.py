"""Compact PyTorch baselines for stock-return prediction."""

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from torch import nn

from ..baselines import regression_preprocessing
from ..preprocessing import TARGET


SEQUENCE_COLUMNS = ("return_1", "intraday_range", "relative_turnover_20")
HISTORY = 20


class ReturnMLP(nn.Module):
    def __init__(self, inputs: int, *, residual: bool):
        super().__init__()
        self.project = nn.Linear(inputs, 32)
        self.hidden = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        self.residual = residual

    def forward(self, x):
        hidden = self.project(x).relu()
        update = self.hidden(hidden).relu()
        return self.output(update + hidden if self.residual else update).squeeze(-1)


class TemporalReturnNetwork(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(2 * len(SEQUENCE_COLUMNS), 8, kernel_size=3), nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, dilation=2), nn.ReLU(),
        )
        self.head = ReturnMLP(inputs + 16, residual=False)

    def forward(self, features, history):
        encoded = self.temporal(history)
        return self.head(torch.cat([features, encoded[:, :, -1], encoded.mean(dim=2)], dim=1))


def _feature_tensors(train, evaluation, feature_columns, device):
    columns = list(feature_columns)
    preprocessing = make_pipeline(*regression_preprocessing(robust=True))
    train_values = preprocessing.fit_transform(train[columns])
    eval_values = preprocessing.transform(evaluation[columns])
    return tuple(torch.as_tensor(values, dtype=torch.float32, device=device) for values in (train_values, eval_values))


def _fit_predict(model, train_batch, eval_batch, target, evaluation_size, loss, *, epochs, batch_size):
    device = target.device
    generator = torch.Generator(device=device).manual_seed(7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    for _ in range(epochs):
        order = torch.randperm(len(target), device=device, generator=generator)
        for batch in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss(model(*train_batch(batch)), target[batch]).backward()
            optimizer.step()

    model.eval()
    with torch.inference_mode():
        batches = torch.arange(evaluation_size, device=device).split(batch_size)
        return torch.cat([model(*eval_batch(batch)) for batch in batches]).cpu().numpy() / 100


def neural_return_scores(
    train: pd.DataFrame, evaluation: pd.DataFrame, feature_columns: tuple[str, ...], *,
    device: str, epochs: int = 4, batch_size: int = 8192, include_huber: bool = False,
) -> pd.DataFrame:
    """Use the same fit-only preprocessing and return loss for both networks."""
    x_train, x_eval = _feature_tensors(train, evaluation, feature_columns, device)
    y_train = torch.as_tensor(train[TARGET].to_numpy() * 100, dtype=torch.float32, device=device)

    variants = [("MLP", False, nn.MSELoss()), ("MLP residual", True, nn.MSELoss())]
    if include_huber:
        variants.append(("MLP Huber", False, nn.HuberLoss(delta=1.0)))
    predictions = {}
    for name, residual, loss in variants:
        torch.manual_seed(7)
        model = ReturnMLP(x_train.shape[1], residual=residual).to(device)
        predictions[name] = _fit_predict(
            model, lambda batch: (x_train[batch],), lambda batch: (x_eval[batch],),
            y_train, len(evaluation), loss, epochs=epochs, batch_size=batch_size,
        )

    return pd.DataFrame(predictions, index=evaluation.index)


def history_batch(panel: torch.Tensor, rows: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Gather windows through the signal day. Pad missing early history with zeros."""
    dates = rows[:, :1] - (len(offsets) - 1) + offsets
    values = panel[dates.clamp_min(0), rows[:, 1:2]]
    return values.masked_fill(dates.unsqueeze(-1).lt(0), 0).transpose(1, 2)


def temporal_return_scores(
    train: pd.DataFrame, evaluation: pd.DataFrame, feature_columns: tuple[str, ...],
    history: pd.DataFrame, *, device: str, epochs: int = 3, batch_size: int = 8192,
) -> pd.DataFrame:
    """Compare chronological and permuted history with a shared signal-day model."""
    x_train, x_eval = _feature_tensors(train, evaluation, feature_columns, device)
    y_train = torch.as_tensor(train[TARGET].to_numpy() * 100, dtype=torch.float32, device=device)
    last_signal = evaluation["Date"].max()
    wide = history.loc[history["Date"].le(last_signal)].set_index(["Date", "SecuritiesCode"])[
        list(SEQUENCE_COLUMNS)
    ].unstack("SecuritiesCode").sort_index()
    dates = wide.index
    codes = wide.columns.levels[1]
    wide = wide.reindex(columns=pd.MultiIndex.from_product([SEQUENCE_COLUMNS, codes]))
    values = wide.to_numpy(dtype=np.float32).reshape(len(dates), len(SEQUENCE_COLUMNS), len(codes)).transpose(0, 2, 1)
    through_train = dates.searchsorted(train["Date"].max(), side="right")
    train_history = values[:through_train].reshape(-1, len(SEQUENCE_COLUMNS))
    low, center, high = np.nanpercentile(train_history, [10, 50, 90], axis=0)
    center = np.nan_to_num(center, nan=0.)
    scale = np.where(high > low, high - low, 1.)
    valid = np.isfinite(values)
    standardized = np.where(valid, np.clip((values - center) / scale, -8., 8.), 0.)
    panel = torch.as_tensor(np.concatenate([standardized, valid], axis=2), dtype=torch.float32, device=device)
    row_indices = []
    for frame in (train, evaluation):
        pairs = np.column_stack([dates.get_indexer(frame["Date"]), codes.get_indexer(frame["SecuritiesCode"])])
        if (pairs < 0).any():
            raise ValueError("Every signal must be present in the sequence calendar and universe.")
        row_indices.append(torch.as_tensor(pairs, dtype=torch.long, device=device))
    train_rows, eval_rows = row_indices

    ordered = np.arange(HISTORY)
    shuffled = np.r_[np.random.default_rng(7).permutation(HISTORY - 1), HISTORY - 1]
    predictions = {}
    for name, order in (("TCN", ordered), ("TCN xáo lịch sử", shuffled)):
        torch.manual_seed(7)
        model = TemporalReturnNetwork(x_train.shape[1]).to(device)
        offsets = torch.as_tensor(order, dtype=torch.long, device=device)
        predictions[name] = _fit_predict(
            model,
            lambda batch: (x_train[batch], history_batch(panel, train_rows[batch], offsets)),
            lambda batch: (x_eval[batch], history_batch(panel, eval_rows[batch], offsets)),
            y_train, len(evaluation), nn.MSELoss(), epochs=epochs, batch_size=batch_size,
        )
    return pd.DataFrame(predictions, index=evaluation.index)
