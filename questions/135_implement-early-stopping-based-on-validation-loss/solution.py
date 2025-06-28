from typing import Tuple


def early_stopping(
    val_losses: list[float], patience: int, min_delta: float
) -> Tuple[int, int]:
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            return epoch, best_epoch

    return len(val_losses) - 1, best_epoch
