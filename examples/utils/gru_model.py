"""GRU-based binary classifier for sequential vital-sign data.

Operates on (N, T, D) arrays of raw per-observation vitals. Missing values
(NaN / zero-padded positions) are communicated to the model via a binary
missingness indicator appended to each time step, giving input shape (N, T, 2D).
Padding is handled correctly via pack_padded_sequence.

Imbalance handling mirrors the XGBoost arm (cv.py): train-fold undersampling
plus odds-space prior correction downstream, and *no* loss reweighting — so the
two model families are corrected identically and per-stratum probabilities stay
comparable. Value channels are standardized using train-fold statistics only
(the mask channel is left as 0/1). Training uses a held-out validation slice for
early stopping, so ``epochs`` is a maximum rather than a fixed budget; this keeps
the model an untuned baseline while avoiding over/under-training that would
confound the per-stratum robustness analysis.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class _GRUNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, static_size: int = 0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + static_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        # x: (N, T, input_size), lengths: (N,), static: (N, static_size) or None
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # h_n: (num_layers, N, hidden_size)
        last_hidden = h_n[-1]  # (N, hidden_size)
        if static is not None:
            last_hidden = torch.cat([last_hidden, static], dim=1)  # (N, hidden_size + static_size)
        return self.fc(self.dropout(last_hidden)).squeeze(1)  # (N,) logits


def _split_value_mask(X_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a (N, T, D) array with NaNs into a NaN-filled value array and mask.

    Returns:
        X_filled: (N, T, D) float32 — vitals with NaN replaced by 0.0
        mask:     (N, T, D) float32 — 1.0 where observed, 0.0 where missing/pad
    """
    mask = (~np.isnan(X_3d)).astype(np.float32)  # (N, T, D)
    X_filled = np.nan_to_num(X_3d, nan=0.0).astype(np.float32)
    return X_filled, mask


def _lengths_from_mask(mask: np.ndarray) -> np.ndarray:
    """Sequence length per stay = last time step with any observed feature.

    Derived from the observed-mask (not value != 0), so it stays correct after
    standardization makes genuinely-observed values equal to 0. Sequences with
    no observed step (rare edge case) are assigned length 1 so packing doesn't
    error.
    """
    any_observed = mask.any(axis=2)  # (N, T)
    lengths = np.zeros(mask.shape[0], dtype=np.int64)
    for i in range(mask.shape[0]):
        idxs = np.flatnonzero(any_observed[i])
        lengths[i] = int(idxs[-1]) + 1 if len(idxs) > 0 else 1
    return lengths


class GRUModel:
    """Binary GRU classifier matching the duck-type interface of XGBoostModel.

    Exposes `_train_model(X_3d, y)` and `_predict(X_3d)` so it can be used as
    a drop-in replacement inside run_cv_gru without touching shared CV utilities.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.05,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 30,
        val_frac: float = 0.1,
        patience: int = 5,
        use_mask: bool = True,
        device: str | None = None,
        random_state: int = 42,
        feature_mode: str = "",
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs  # maximum epochs; early stopping may stop sooner
        self.val_frac = val_frac
        self.patience = patience
        # When False, the binary observed-mask channel is dropped from the input
        # (values only, zero-filled at missing positions). Used for the mask
        # ablation: comparing use_mask True vs False per stratum quantifies how
        # much information the missingness indicator adds where.
        self.use_mask = use_mask
        self.random_state = random_state
        self.feature_mode = feature_mode

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: _GRUNet | None = None
        # Per-feature standardization stats, fit on train-fold observed values.
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        # Static feature standardization stats (mean/std per column).
        self._static_mean: np.ndarray | None = None
        self._static_std: np.ndarray | None = None

    def _fit_scaler(self, X_filled: np.ndarray, mask: np.ndarray) -> None:
        """Fit per-feature mean/std over observed (mask=1) train values only.

        Padded/missing positions are excluded so the statistics reflect real
        readings, not the zero fill. Stored on the instance and reused at predict
        time — each CV fold uses a fresh GRUModel, so there is no leakage.
        """
        D = X_filled.shape[2]
        mean = np.zeros(D, dtype=np.float32)
        std = np.ones(D, dtype=np.float32)
        for d in range(D):
            observed = X_filled[:, :, d][mask[:, :, d] > 0]
            if observed.size > 0:
                mean[d] = float(observed.mean())
                s = float(observed.std())
                std[d] = s if s > 1e-8 else 1.0
        self._feat_mean = mean
        self._feat_std = std

    def _build_input(self, X_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, T, D) with NaNs -> standardized model input + lengths.

        Value channels are standardized with the fitted train stats; missing
        positions are kept at 0 *after* standardization so padding stays zero.
        When ``use_mask`` is True the binary observed-mask is appended, giving
        (N, T, 2D); otherwise the input is values-only (N, T, D) — the mask
        ablation. Lengths are always derived from the mask, independent of
        ``use_mask``, so padding is packed correctly in both arms.
        """
        if self._feat_mean is None or self._feat_std is None:
            raise RuntimeError("Scaler not fit; call _train_model first.")

        X_filled, mask = _split_value_mask(X_3d)
        lengths = _lengths_from_mask(mask)

        X_scaled = (X_filled - self._feat_mean) / self._feat_std
        X_scaled = X_scaled * mask  # missing/padded positions back to 0

        if self.use_mask:
            X_combined = np.concatenate([X_scaled, mask], axis=2).astype(np.float32)  # (N, T, 2D)
        else:
            X_combined = X_scaled.astype(np.float32)  # (N, T, D)
        return X_combined, lengths

    def _fit_static_scaler(self, X_static: np.ndarray) -> None:
        """Fit per-column mean/std over observed (non-NaN) train static values."""
        S = X_static.shape[1]
        mean = np.zeros(S, dtype=np.float32)
        std = np.ones(S, dtype=np.float32)
        for s in range(S):
            observed = X_static[:, s][~np.isnan(X_static[:, s])]
            if observed.size > 0:
                mean[s] = float(observed.mean())
                sc = float(observed.std())
                std[s] = sc if sc > 1e-8 else 1.0
        self._static_mean = mean
        self._static_std = std

    def _scale_static(self, X_static: np.ndarray) -> np.ndarray:
        """Standardize and zero-fill NaN static features using fitted stats."""
        if self._static_mean is None or self._static_std is None:
            raise RuntimeError("Static scaler not fit; call _train_model first.")
        X_scaled = (X_static - self._static_mean) / self._static_std
        return np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

    def _stratified_val_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index split into (train, val) keeping the class ratio in both parts."""
        rng = np.random.default_rng(self.random_state)
        train_idx_parts, val_idx_parts = [], []
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls)
            rng.shuffle(cls_idx)
            n_val = max(1, int(round(len(cls_idx) * self.val_frac))) if len(cls_idx) > 1 else 0
            val_idx_parts.append(cls_idx[:n_val])
            train_idx_parts.append(cls_idx[n_val:])
        train_idx = np.concatenate(train_idx_parts)
        val_idx = (
            np.concatenate(val_idx_parts) if any(len(p) for p in val_idx_parts) else np.array([], dtype=int)
        )
        return train_idx, val_idx

    def _train_model(self, X_3d: np.ndarray, y: np.ndarray, X_static: np.ndarray | None = None) -> None:
        """Train the GRU on a (N, T, D) array and binary labels y (N,).

        X_static: optional (N, S) array of static features (e.g. age) that are
        concatenated to the last GRU hidden state before the classifier head.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_filled, mask = _split_value_mask(X_3d)
        self._fit_scaler(X_filled, mask)
        X_combined, lengths = self._build_input(X_3d)
        input_size = X_combined.shape[2]

        static_size = 0
        X_static_scaled: np.ndarray | None = None
        if X_static is not None:
            self._fit_static_scaler(X_static)
            X_static_scaled = self._scale_static(X_static)
            static_size = X_static_scaled.shape[1]

        y_arr = np.asarray(y, dtype=np.float32)

        net = _GRUNet(input_size, self.hidden_size, self.num_layers, self.dropout, static_size).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        tr_idx, val_idx = self._stratified_val_split(y_arr)
        has_val = len(val_idx) > 0
        if has_val:
            xv = torch.tensor(X_combined[val_idx], device=self.device)
            yv = torch.tensor(y_arr[val_idx], device=self.device)
            lv = torch.tensor(lengths[val_idx], device=self.device)
            sv = torch.tensor(X_static_scaled[val_idx], device=self.device) if X_static_scaled is not None else None

        N_tr = len(tr_idx)
        rng = np.random.default_rng(self.random_state)

        best_val = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        for _ in range(self.epochs):
            net.train()
            perm = tr_idx[rng.permutation(N_tr)]
            for start in range(0, N_tr, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = torch.tensor(X_combined[idx], device=self.device)
                yb = torch.tensor(y_arr[idx], device=self.device)
                lb = torch.tensor(lengths[idx], device=self.device)
                sb = torch.tensor(X_static_scaled[idx], device=self.device) if X_static_scaled is not None else None

                optimizer.zero_grad()
                logits = net(xb, lb, sb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if not has_val:
                continue

            net.eval()
            with torch.no_grad():
                val_loss = float(criterion(net(xv, lv, sv), yv).item())
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net

    def _predict(self, X_3d: np.ndarray, X_static: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_pred, y_proba) for a (N, T, D) array.

        X_static: optional (N, S) static features — must match what was passed
        to _train_model. Scaled with the fitted static scaler.
        """
        if self.model is None:
            raise RuntimeError("Call _train_model before _predict.")

        X_combined, lengths = self._build_input(X_3d)
        X_static_scaled: np.ndarray | None = None
        if X_static is not None:
            X_static_scaled = self._scale_static(X_static)
        N = X_combined.shape[0]

        self.model.eval()
        all_proba: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, N, self.batch_size):
                xb = torch.tensor(X_combined[start : start + self.batch_size], device=self.device)
                lb = torch.tensor(lengths[start : start + self.batch_size], device=self.device)
                sb = torch.tensor(X_static_scaled[start : start + self.batch_size], device=self.device) if X_static_scaled is not None else None
                logits = self.model(xb, lb, sb)
                proba = torch.sigmoid(logits).cpu().numpy()
                all_proba.append(proba)

        y_proba = np.concatenate(all_proba, axis=0)
        y_pred = (y_proba >= 0.5).astype(np.int8)
        return y_pred, y_proba
