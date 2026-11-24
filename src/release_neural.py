"""Train and reload equally budgeted tabular and sequence neural baselines."""

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from torch import nn

from .baselines import regression_preprocessing
from .models.neural import HISTORY, SEQUENCE_COLUMNS, ReturnMLP, TemporalReturnNetwork, history_batch
from .preprocessing import KEY, TARGET

SEEDS = (7, 19, 37)
BATCH = 8192


class GRUReturnNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.gru = nn.GRU(2 * len(SEQUENCE_COLUMNS), 32, batch_first=True)
        self.head = ReturnMLP(size + 32, residual=False)

    def forward(self, static, sequence):
        _, hidden = self.gru(sequence.transpose(1, 2).contiguous())
        return self.head(torch.cat([static, hidden[-1]], dim=1))


def _model(family, size):
    if family == "mlp":
        return ReturnMLP(size, residual=False)
    return (GRUReturnNetwork if family == "gru" else TemporalReturnNetwork)(size)


def _matrix(history):
    wide = history.set_index(KEY)[list(SEQUENCE_COLUMNS)].unstack().sort_index()
    codes = wide.columns.levels[1]
    wide = wide.reindex(columns=pd.MultiIndex.from_product([SEQUENCE_COLUMNS, codes]))
    values = wide.to_numpy(dtype=np.float32).reshape(len(wide), len(SEQUENCE_COLUMNS), len(codes))
    return wide.index, codes, values.transpose(0, 2, 1)


def _batch_source(bundle, rows, history, device):
    static = torch.as_tensor(bundle["preprocessing"].transform(rows[bundle["columns"]]),
                             dtype=torch.float32, device=device)
    if bundle["family"] == "mlp":
        return lambda batch: (static[batch],)
    dates, codes, values = _matrix(history.loc[history["Date"].le(rows["Date"].max())])
    center, scale = bundle["sequence_scale"]
    valid = np.isfinite(values)
    x = np.where(valid, np.clip((values - center) / scale, -8, 8), 0).astype(np.float32)
    tensor = torch.as_tensor(np.concatenate([x, valid], axis=2), device=device)
    pairs = np.column_stack([dates.get_indexer(rows["Date"]), codes.get_indexer(rows["SecuritiesCode"])])
    if (pairs < 0).any():
        raise ValueError("Signal missing from sequence panel.")
    pairs = torch.as_tensor(pairs, device=device)
    offsets = np.arange(HISTORY)
    if bundle["family"] == "tcn_shuffled":
        offsets[:-1] = np.random.default_rng(31).permutation(HISTORY - 1)
    offsets = torch.as_tensor(offsets, device=device)
    return lambda batch: (static[batch], history_batch(tensor, pairs[batch], offsets))


def fit_neural(family, fit, columns, history, device):
    prep = make_pipeline(*regression_preprocessing(robust=True)).fit(fit[columns])
    bundle = {"family": family, "columns": columns, "preprocessing": prep,
              "width": prep.transform(fit[columns].iloc[:1]).shape[1], "states": {}, "learning_curve": []}
    if family != "mlp":
        _, _, raw = _matrix(history.loc[history["Date"].le(fit["Date"].max())])
        low, center, high = np.nanpercentile(raw.reshape(-1, len(SEQUENCE_COLUMNS)), [10, 50, 90], axis=0)
        bundle["sequence_scale"] = (np.nan_to_num(center), np.where(high > low, high - low, 1.))
    batches = _batch_source(bundle, fit, history, device)
    target = torch.as_tensor(fit[TARGET].to_numpy() * 100, dtype=torch.float32, device=device)
    for seed in SEEDS:
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        model = _model(family, bundle["width"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        for epoch in range(1, 5):
            loss_total = torch.zeros((), device=device)
            for batch in torch.randperm(len(fit), generator=generator, device=device).split(BATCH):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(*batches(batch)), target[batch])
                loss.backward()
                optimizer.step()
                loss_total += loss.detach() * len(batch)
            bundle["learning_curve"].append({"Seed": seed, "Epoch": epoch,
                                             "Training MSE percent squared": loss_total.item() / len(fit)})
        bundle["states"][seed] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return bundle


def predict_neural(bundle, signals, history, device):
    batches = _batch_source(bundle, signals, history, device)
    predictions = {}
    with torch.inference_mode():
        for seed, state in bundle["states"].items():
            model = _model(bundle["family"], bundle["width"]).to(device).eval()
            model.load_state_dict(state)
            batch_ids = torch.arange(len(signals), device=device).split(BATCH)
            predictions[seed] = torch.cat([model(*batches(ids)) for ids in batch_ids]).cpu().numpy() / 100
    return pd.DataFrame(predictions, index=signals.index)
