import numpy as np
import torch
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


def compute_det_curve(target_scores, nontarget_scores):
    target_scores = np.asarray(target_scores, dtype=np.float64).ravel()
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64).ravel()

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(bonafide_scores, other_scores):
    bonafide_scores = np.asarray(bonafide_scores, dtype=np.float64).ravel()
    other_scores = np.asarray(other_scores, dtype=np.float64).ravel()
    if bonafide_scores.size == 0 or other_scores.size == 0:
        raise ValueError(
            f"Empty score arrays: bona={bonafide_scores.size}, spoof={other_scores.size}"
        )

    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = float(np.mean((frr[min_index], far[min_index])))

    if not (0.0 <= eer <= 1.0):
        raise RuntimeError(
            f"Invalid EER={eer} (expected in [0, 1]). Check scores/labels."
        )
    return eer, thresholds[min_index]


class EERMetric(BaseMetric):
    def __init__(self, name="EER", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.scores = []
        self.labels = []

    def reset(self):
        self.scores = []
        self.labels = []

    def __call__(self, logits, labels, **kwargs):
        probs = F.softmax(logits, dim=-1)
        bona_scores = probs[:, 0].detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        self.scores.extend(bona_scores.tolist())
        self.labels.extend(labels_np.tolist())
        return 0.0

    def finalize(self):
        scores = np.array(self.scores, dtype=np.float64)
        labels_arr = np.array(self.labels, dtype=np.int64)
        eer, _ = compute_eer(scores[labels_arr == 0], scores[labels_arr == 1])
        return eer
