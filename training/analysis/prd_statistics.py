"""Offline statistics for PRD residual feature archives."""

import numpy as np


def rbf_mmd(first, second):
    """Compute biased RBF-kernel MMD with a median-distance bandwidth."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    pooled = np.concatenate([first, second], axis=0)
    squared_norm = (pooled ** 2).sum(axis=1, keepdims=True)
    squared = np.maximum(squared_norm + squared_norm.T - 2.0 * pooled @ pooled.T, 0.0)
    bandwidth = np.median(squared[squared > 0]) or 1.0
    kernel = np.exp(-squared / (2.0 * bandwidth))
    count_first = len(first)
    return float(kernel[:count_first, :count_first].mean() + kernel[count_first:, count_first:].mean() - 2.0 * kernel[:count_first, count_first:].mean())


def class_statistics(residual, labels):
    """Report real/fake residual norms, standardized mean gap, and RBF-MMD."""
    residual, labels = np.asarray(residual), np.asarray(labels)
    real, fake = residual[labels == 0], residual[labels == 1]
    if not len(real) or not len(fake):
        raise ValueError("Both real and fake residual samples are required for class statistics.")
    real_norm, fake_norm = np.linalg.norm(real, axis=1), np.linalg.norm(fake, axis=1)
    pooled_std = np.sqrt((real_norm.var() + fake_norm.var()) / 2.0)
    effect_size = float((fake_norm.mean() - real_norm.mean()) / (pooled_std + 1e-8))
    return {"mean_residual_norm_real": float(real_norm.mean()), "mean_residual_norm_fake": float(fake_norm.mean()), "effect_size_cohens_d": effect_size, "class_sep_mmd_rbf": rbf_mmd(real, fake)}


def domain_statistics(train_residual, train_labels, test_residual, test_labels):
    """Compute label-conditional domain MMD and the requested exploratory score."""
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    real = rbf_mmd(train_residual[train_labels == 0], test_residual[test_labels == 0])
    fake = rbf_mmd(train_residual[train_labels == 1], test_residual[test_labels == 1])
    domain_sep = 0.5 * (real + fake)
    class_sep = class_statistics(train_residual, train_labels)["class_sep_mmd_rbf"]
    return {"domain_shift_real_mmd_rbf": real, "domain_shift_fake_mmd_rbf": fake, "class_sep_mmd_rbf": class_sep, "domain_sep_mmd_rbf": domain_sep, "exploratory_score": class_sep / (domain_sep + 1e-8)}
