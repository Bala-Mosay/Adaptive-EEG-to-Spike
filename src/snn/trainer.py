"""Training loop for the baseline SNN (Stage 6, baseline version).

Baseline loss is CrossEntropy ONLY — no spike-count regularisation
(``lambda_spike = 0``). The joint CE + sparsity + entropy objective
(Eq. 10/13 of the research paper) is Phase 3 Contribution 2.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from src.snn.encoder import FixedThresholdEncoder


@dataclass
class TrainingConfig:
    """All hyperparameters for baseline SNN training.

    Attributes mirror ``config/phase2_config.yaml`` (training section).
    """

    n_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    T: int = 16  # timesteps
    lambda_spike: float = 0.0  # NO spike regularisation in baseline (Phase 3 adds it)
    max_grad_norm: float = 1.0
    device: str = "cpu"
    random_state: int = 42


class BaselineTrainer:
    """Deterministic training loop for :class:`~src.snn.lif_net.BaselineLIFNet`.

    Uses Adam + cosine-annealed LR, gradient-norm clipping, and ATan surrogate
    gradients (handled inside SpikingJelly's LIFNode). Batch shuffling is
    seeded from ``config.random_state`` for reproducibility.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig,
                 encoder: Optional[FixedThresholdEncoder] = None):
        self.model = model.to(config.device)
        self.config = config
        self.encoder = encoder
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs,
        )
        self._rng = np.random.default_rng(config.random_state)

    def _encode(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Convert features to binary spikes and repeat across T timesteps.

        If an encoder is provided, applies threshold-based binary encoding
        (Stage 4 of the paper). Otherwise, repeats raw features (for unit
        tests only).
        """
        if self.encoder is not None:
            spike = self.encoder.encode_tensor(x_batch)
        else:
            spike = (x_batch > 0).float()
        return spike.unsqueeze(0).repeat(self.config.T, 1, 1)

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train for one epoch; return the mean loss dict.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape ``(n_trials, n_features)``.
        y : np.ndarray
            Integer labels, shape ``(n_trials,)``.
        """
        self.model.train()
        device = self.config.device
        indices = self._rng.permutation(len(X))
        bs = self.config.batch_size
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(X), bs):
            batch_idx = indices[start:start + bs]
            x_batch = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=device)
            y_batch = torch.as_tensor(y[batch_idx], dtype=torch.long, device=device)

            self.optimizer.zero_grad()
            logits, _ = self.model(self._encode(x_batch))

            # Baseline loss: CE only, no spike regularisation.
            loss = self.criterion(logits, y_batch)
            # NOTE (Phase 3): loss += lambda(e) * spike_count

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return {"ce_loss": total_loss / max(n_batches, 1)}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
        """Evaluate without gradients; return predictions and spike counts.

        Returns
        -------
        dict with keys ``predictions`` (np.ndarray of int) and
        ``total_spike_count`` (float).
        """
        self.model.eval()
        device = self.config.device
        all_preds: List[np.ndarray] = []
        all_spike_counts: List[float] = []

        with torch.no_grad():
            bs = self.config.batch_size
            for start in range(0, len(X), bs):
                x_batch = torch.as_tensor(X[start:start + bs], dtype=torch.float32, device=device)
                logits, spike_count = self.model(self._encode(x_batch))
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_spike_counts.append(float(spike_count.item()))

        return {
            "predictions": np.concatenate(all_preds),
            "total_spike_count": sum(all_spike_counts),
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> List[Dict[str, float]]:
        """Full training run; returns the per-epoch history.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training features and integer labels.
        X_val, y_val : np.ndarray, optional
            Optional validation set for per-epoch accuracy tracking.
        verbose : bool
            Print progress every 10 epochs.
        """
        history: List[Dict[str, float]] = []
        for epoch in range(self.config.n_epochs):
            train_metrics = self.train_epoch(X_train, y_train)
            record: Dict[str, float] = {"epoch": epoch, **train_metrics}
            if X_val is not None:
                val_out = self.evaluate(X_val, y_val)
                record["val_acc"] = float(np.mean(val_out["predictions"] == y_val))
            history.append(record)
            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch:3d} | CE={record['ce_loss']:.4f}"
                if X_val is not None:
                    msg += f" | Val={record.get('val_acc', 0.0):.3f}"
                print(msg)
        return history
