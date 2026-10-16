import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_metrics(y_true, y_pred, n_classes):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n_classes)),
                                                  average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f1),
            "confusion": confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))}


def delta(acc_train, acc_r):
    """ΔA(R) = A(R_train) - A(R) y ΔA%(R) (ecs. 4.18-4.19 de la tesis)."""
    d = acc_train - acc_r
    return d, (d / acc_train * 100 if acc_train > 0 else float("nan"))
