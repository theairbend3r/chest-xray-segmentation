import torch


def iou_score(y_pred, y_true):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1).sum(dim=-1) + y_pred.sum(dim=-2).sum(
        dim=-1
    ).sum(dim=-1)

    return (intersection / (union - intersection + epsilon)).mean()


# computes confusion matrix
# https://github.com/kevinzakka/pytorch-goodies
EPS = 1e-6


def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = (
        torch.bincount(
            num_classes * true[mask] + pred[mask], minlength=num_classes ** 2,
        )
        .reshape(num_classes, num_classes)
        .float()
    )
    return hist


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


# computes IoU based on confusion matrix
def jaccard_index(true, pred, num_classes):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    hist = _fast_hist(true, pred, num_classes)

    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)  # the mean of jaccard without NaNs

    return avg_jacc, jaccard


def dice_coef(true, pred, num_classes):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    hist = _fast_hist(true, pred, num_classes)

    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)

    return avg_dice
