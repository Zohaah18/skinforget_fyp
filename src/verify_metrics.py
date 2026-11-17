"""Pure numpy/sklearn metrics for the verification suite (no torch needed, easy to test)."""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def sample_stats(logits, y):
    """Per-sample cross-entropy loss (true label), max softmax confidence, prediction entropy."""
    lp = log_softmax(logits.astype(np.float64))
    p = np.exp(lp)
    loss = np.minimum(-lp[np.arange(len(y)), y], 50.0)
    return loss, p.max(1), -(p * lp).sum(1)


def tpr_at_fpr(y_true, score, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.interp(fpr_target, fpr, tpr))


def mia(stats_member, stats_nonmember, seed=0):
    """Membership inference: members = forgotten training images, non-members = held-out images
    of the same classes. Single-feature threshold attacks + a 5-fold-CV logistic regression on
    (loss, confidence, entropy). Returns AUCs and TPR at 1% FPR."""
    X = np.vstack([np.column_stack(stats_member), np.column_stack(stats_nonmember)])
    y = np.r_[np.ones(len(stats_member[0])), np.zeros(len(stats_nonmember[0]))]

    def two_sided(a):                                   # direction-free: 0.5 = chance, 1.0 = perfect
        return float(max(a, 1.0 - a))
    out = {"auc_loss": two_sided(roc_auc_score(y, -X[:, 0])),
           "auc_conf": two_sided(roc_auc_score(y, X[:, 1])),
           "auc_ent": two_sided(roc_auc_score(y, -X[:, 2]))}
    F = np.column_stack([np.log1p(X[:, 0]), X[:, 1], np.log1p(X[:, 2])])
    k = int(min(5, (y == 1).sum(), (y == 0).sum()))
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(F, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500))
        clf.fit(F[tr], y[tr])
        oof[te] = clf.decision_function(F[te])
    out["auc_lr"] = roc_auc_score(y, oof)
    out["tpr1_lr"] = tpr_at_fpr(y, oof, 0.01)
    return out


def linear_probe(F_train, y_train, F_test, y_test, C=0.1):
    """Can 'forgotten class vs the rest' still be linearly read out of the frozen features?"""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=300))
    clf.fit(F_train, y_train)
    return {"bal_acc": float(balanced_accuracy_score(y_test, clf.predict(F_test))),
            "auc": float(roc_auc_score(y_test, clf.decision_function(F_test)))}


def kl_to_gold(p_gold, p_model, eps=1e-8):
    p, q = np.clip(p_gold, eps, 1.0), np.clip(p_model, eps, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(1)