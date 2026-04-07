# fincdm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import re

Array = np.ndarray


def snmcf(
    X: Array, Q: Array, r: int = 38, alpha: float = 1.0,
    lambda_E: float = 0.0, lambda_U: float = 0.0, lambda_V: float = 0.0,
    max_iter: int = 200, tol: float = 1e-4, seed: int = 42, verbose: bool = False
) -> Tuple[Array, Array, Array]:
    rng = np.random.default_rng(seed)
    N, M = X.shape
    Nq, K = Q.shape
    assert N == Nq, "X and Q must have the same number of exercises (rows)."

    W = (~np.isnan(X)).astype(float)
    X_filled = np.nan_to_num(X, nan=0.0)

    E = rng.random((N, r)) + 1e-2
    U = rng.random((r, M)) + 1e-2
    V = rng.random((r, K)) + 1e-2
    eps = 1e-10

    def loss() -> float:
        term1 = np.linalg.norm((W * (X_filled - E @ U))) ** 2
        term2 = alpha * np.linalg.norm(Q - E @ V) ** 2
        reg = (lambda_E * np.linalg.norm(E) ** 2
               + lambda_U * np.linalg.norm(U) ** 2
               + lambda_V * np.linalg.norm(V) ** 2)
        return term1 + term2 + reg

    prev = loss()
    if verbose:
        print(f"Iter 0\tLoss = {prev:.4e}")

    for it in range(1, max_iter + 1):
        numer_E = (W * X_filled) @ U.T + alpha * Q @ V.T
        denom_E = (W * (E @ U)) @ U.T + alpha * E @ (V @ V.T) + lambda_E * E + eps
        E *= numer_E / denom_E

        numer_U = E.T @ (W * X_filled)
        denom_U = E.T @ (W * (E @ U)) + lambda_U * U + eps
        U *= numer_U / denom_U

        numer_V = E.T @ Q
        denom_V = E.T @ E @ V + lambda_V * V + eps
        V *= numer_V / denom_V

        if it % 10 == 0 or it == max_iter:
            curr = loss()
            rel = abs(prev - curr) / (prev + eps)
            if verbose:
                print(f"Iter {it}\tLoss = {curr:.4e}\tΔ={rel:.2e}")
            if rel < tol:
                break
            prev = curr

    return E, U, V

def predict_scores(E: Array, U: Array) -> Array:
    return E @ U

def student_knowledge(U: Array, V: Array) -> Array:
    return U.T @ V

def _binarise(x: Array, thr: float = 0.5) -> Array:
    return (x >= thr).astype(int)

def compute_metrics(X_true: Array, X_pred: Array, thr: float = 0.5) -> Dict[str, float]:
    mask = ~np.isnan(X_true)
    y_true = X_true[mask]
    y_pred = X_pred[mask]

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    if set(np.unique(y_true)) <= {0, 1}:
        acc = float(np.mean(_binarise(y_pred, thr) == y_true))
    else:
        acc = np.nan

    auc = np.nan
    if acc is not np.nan:
        pos = y_true == 1
        neg = y_true == 0
        if pos.any() and neg.any():
            try:
                from sklearn.metrics import roc_auc_score  # type: ignore
                auc = float(roc_auc_score(y_true, y_pred))
            except Exception:
                rank = np.argsort(np.argsort(y_pred))
                sum_ranks_pos = float(np.sum(rank[pos]))
                n_pos = int(pos.sum())
                n_neg = int(neg.sum())
                auc = (sum_ranks_pos - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
        else:
            auc = np.nan

    return {"ACC": acc, "AUC": auc, "MSE": mse, "RMSE": rmse}


@dataclass
class EvalResults:
    dataset: str
    y_true: Array
    y_pred: Array
    metrics: Dict[str, float]
    meta: Dict[str, Any]
    aux: Dict[str, Any]



class FinCDMEvaluator:

    def __init__(self, data_root: str = ".", snmcf_rank: int = 38):
        self.data_root = data_root
        self.snmcf_rank = snmcf_rank

    def evaluate(
        self,
        dataset: str = "cpa-kqa",
        *,
        q_path: Optional[str] = None,
        a_path: Optional[str] = None,
        binarise_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> EvalResults:
        R_df, Q_df = self._load_dataset(dataset, q_path=q_path, a_path=a_path)
        R = R_df.to_numpy(float)  # (m, n)
        Q = Q_df.to_numpy(float)  # (m, k)

 
        E, U, V = snmcf(R, Q, r=self.snmcf_rank, alpha=1.0, max_iter=200, verbose=verbose)
        y_pred = predict_scores(E, U)
        aux = {"E": E, "U": U, "V": V}


        if binarise_threshold is not None:
            y_pred = np.clip(y_pred, 0.0, 1.0)

        metrics = compute_metrics(R, y_pred, thr=binarise_threshold or 0.5)

        aux.update({"R": R, "Q": Q, "R_df": R_df, "Q_df": Q_df})
        return EvalResults(
            dataset=dataset,
            y_true=R,
            y_pred=y_pred,
            metrics=metrics,
            meta={
                "student_names": R_df.columns.tolist(),
                "concept_names": Q_df.columns.tolist(),
                "item_ids": R_df.index.tolist(),
            },
            aux=aux,
        )

    def diagnose(
        self,
        results: EvalResults,
        *,
        average_trials: bool = True,
        export_csv: Optional[str] = "SK_df.csv",
    ) -> Dict[str, Any]:

        if all(k in results.aux for k in ("E", "U", "V")):
            U, V = results.aux["U"], results.aux["V"]
        else:
            R, Q = results.aux["R"], results.aux["Q"]
            _, U, V = snmcf(R, Q, r=self.snmcf_rank, alpha=1.0, max_iter=200, verbose=False)

        R_df: pd.DataFrame = results.aux["R_df"]
        Q_df: pd.DataFrame = results.aux["Q_df"]

        SK = student_knowledge(U, V)  # (n_students, n_concepts)
        SK_df = pd.DataFrame(SK, index=R_df.columns.tolist(), columns=Q_df.columns.tolist())

        if average_trials:
            SK_df = self._average_trials(SK_df)

        if export_csv:
            SK_df.to_csv(export_csv, index=True, encoding="utf-8-sig")

        return {"SK_df": SK_df, "U": U, "V": V}


    def _load_dataset(
        self, name: str, *, q_path: Optional[str], a_path: Optional[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if q_path and a_path:
            Q_df = pd.read_excel(q_path).set_index("题号")
            R_df = pd.read_excel(a_path).set_index("题号")
        elif name.lower() == "cpa-kqa":
            Q_df = pd.read_excel(f"{self.data_root}/cpa_new_q.xlsx").set_index("题号")
            R_df = pd.read_excel(f"{self.data_root}/cpa_new_a.xlsx").set_index("题号")
        else:
            raise ValueError(f"Unknown dataset: {name}")

        common = Q_df.index.intersection(R_df.index)
        Q_df = Q_df.loc[common]
        R_df = R_df.loc[common]
        return R_df, Q_df

    def _predict_with_fallback(self, model: Any, R: Array, Q: Array) -> Array:
        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            y_pred = model.predict(R, Q)
        elif callable(model):
            y_pred = model(R, Q)

        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.shape != R.shape:
            raise ValueError(f"Prediction shape {y_pred.shape} != labels shape {R.shape}")
        return y_pred

    def _average_trials(self, SK_df: pd.DataFrame) -> pd.DataFrame:
        def extract_model_name(student_name: str) -> str:
            m = re.match(r'(.*?)-Trial\d', student_name)
            return m.group(1) if m else student_name

        groups: Dict[str, list] = {}
        for s in SK_df.index:
            groups.setdefault(extract_model_name(s), []).append(s)

        averaged = {name: SK_df.loc[rows].mean(axis=0) for name, rows in groups.items()}
        return pd.DataFrame(averaged).T



