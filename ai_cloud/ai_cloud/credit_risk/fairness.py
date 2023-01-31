# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Utility functions for evaluating fairness metrics.
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import xgboost as xgb


def get_predictive_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Compute the predictive metrics given the confusion matrix values.

    Args:
        tp (int): true positives
        fp (int): false positives
        fn (int): false negatives
        tn (int): true negatives

    Returns:
        Dict[str, float]: Dictionary of computed metrics
    """

    ppv = tp/(tp + fp)
    fdr = fp/(tp + fp)
    npv = tn/(tn + fn)
    fomr = fn/(tn + fn)
    tpr = tp/(tp + fn)
    fnr = fn/(tp + fn)
    tnr = tn/(tn + fp)
    fpr = fp/(tn + fp)
    return {
        "ppv": ppv,
        "fdr": fdr,
        "npv": npv,
        "fomr": fomr,
        "tpr": tpr,
        "fnr": fnr,
        "tnr": tnr,
        "fpr": fpr
    }

def get_fairness_parity_report(
        model: xgb.core.Booster,
        X: pd.DataFrame,
        y_true: np.ndarray,
        privilege_indicator: np.ndarray) -> Dict[str, float]:
    """
    Generate a report of model fairness for a binary classification model
    using parity of predictive values.

    Args:
        model (xgb.core.Booster): trained xgb to evaluate
        X (pd.DataFrame): dataset to evaluate predictions
        y_true (np.ndarray): true labels
        privilege_indicator (np.ndarray): indicator array for whether each row is privileged

    Returns:
        Dict[str, float]:  Dictionary of parity of predictive metrics.
    """

    y_pred = (model.predict(xgb.DMatrix(X)) > 0.5).astype(int)

    # predictive metrics of privileged group
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(
        y_true[privilege_indicator == 1],
        y_pred[privilege_indicator == 1]
    ).ravel()

    pred_p = get_predictive_metrics(
        tp_p, fp_p, fn_p, tn_p
    )

    # predictive metrics of non-privileged group
    tn_np, fp_np, fn_np, tp_np = confusion_matrix(
        y_true[privilege_indicator == 0],
        y_pred[privilege_indicator == 0]
    ).ravel()

    pred_np = get_predictive_metrics(
        tp_np, fp_np, fn_np, tn_np
    )

    return {
        "ppv": pred_p["ppv"]/pred_np["ppv"],
        "fdr": pred_p["fdr"]/pred_np["fdr"],
        "npv": pred_p["npv"]/pred_np["npv"],
        "fomr": pred_p["fomr"]/pred_np["fomr"],
        "tpr": pred_p["tpr"]/pred_np["tpr"],
        "fnr": pred_p["fnr"]/pred_np["fnr"],
        "tnr": pred_p["tnr"]/pred_np["tnr"],
        "fpr": pred_p["fpr"]/pred_np["fpr"]
    }

__all__ = ["get_fairness_parity_report"]
