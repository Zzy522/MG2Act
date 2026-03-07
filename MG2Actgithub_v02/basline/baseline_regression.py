"""
回归基线：RandomForest / SVR / XGBoost + Morgan & MACCS 指纹 + 蛋白 ACC
（auto cross-covariance，自协/互协）特征

评估指标：MAE / RMSE / R²
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import random
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import RDLogger
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: idx for idx, aa in enumerate(AA_LIST)}

def load_split(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = ["Molecule_SMILES", "E3_seq", "Target_seq", "Score"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少必要列: {missing}")
    return df.dropna(subset=cols).reset_index(drop=True)

def smiles_to_morgan(smiles: str, radius: int = 3, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def smiles_to_maccs(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(166, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)  # 167 bits，索引0通常固定为0
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    # 去掉第0位，与常见实现保持一致
    arr = arr[1:]
    return arr.astype(np.float32)

def sequence_to_acc_cov(seq: str, max_lag: int = 3) -> np.ndarray:
    """
    计算 auto cross-covariance (ACC) 特征。
    - 使用 20 维 one-hot 表示
    - 特征 = 均值(20) + 每个 lag 的 20x20 协方差展开（总维度 20 + max_lag*400）
    """
    dim = len(AA_LIST)
    if not seq:
        return np.zeros(20 + max_lag * dim * dim, dtype=np.float32)
    onehots = []
    for ch in seq:
        idx = AA_INDEX.get(ch)
        vec = np.zeros(dim, dtype=np.float32)
        if idx is not None:
            vec[idx] = 1.0
        onehots.append(vec)
    if not onehots:
        return np.zeros(20 + max_lag * dim * dim, dtype=np.float32)
    X = np.stack(onehots)  # [L, 20]
    mean = X.mean(axis=0, keepdims=True)  # [1, 20]
    feats = [mean.ravel()]
    for lag in range(1, max_lag + 1):
        if X.shape[0] <= lag:
            feats.append(np.zeros(dim * dim, dtype=np.float32))
            continue
        x1 = X[:-lag] - mean
        x2 = X[lag:] - mean
        cov = np.einsum("ni,nj->ij", x1, x2) / (x1.shape[0])
        feats.append(cov.astype(np.float32).ravel())
    return np.concatenate(feats, axis=0)

def protein_feature(e3_seq: str, target_seq: str) -> np.ndarray:
    """使用 ACC（自协/互协）特征编码蛋白质序列"""
    return np.concatenate(
        [
            sequence_to_acc_cov(e3_seq),
            sequence_to_acc_cov(target_seq),
        ]
    )

def build_feature_components(df, mol_feature: str, *, radius: int = 3, n_bits: int = 2048):
    mol_feats = []
    protein_feats = []

    for _, row in df.iterrows():
        smiles = row["Molecule_SMILES"]
        if mol_feature == "morgan":
            mol_vec = smiles_to_morgan(smiles, radius=radius, n_bits=n_bits)
        elif mol_feature == "maccs":
            mol_vec = smiles_to_maccs(smiles)
        else:
            raise ValueError(f"未知分子特征类型: {mol_feature}")
        mol_feats.append(mol_vec)
        protein_feats.append(protein_feature(row["E3_seq"], row["Target_seq"]))

    mol_arr = np.stack(mol_feats).astype(np.float32)
    prot_arr = np.stack(protein_feats).astype(np.float32)
    return mol_arr, prot_arr

def apply_linear_projection(features, input_dim: int, output_dim: int, *, random_state: int):
    """使用固定随机矩阵进行线性投影降维"""
    np.random.seed(random_state)
    # 生成固定随机正交矩阵用于投影
    projection_matrix = np.random.randn(input_dim, output_dim).astype(np.float32)
    # 正交化
    u, _, vt = np.linalg.svd(projection_matrix, full_matrices=False)
    projection_matrix = u @ vt
    return features @ projection_matrix

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class EvalResult:
    name: str
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    test_true: np.ndarray | None = None
    test_pred: np.ndarray | None = None


def make_regressor(kind: str, random_state: int) -> RegressorMixin:
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if kind == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(kernel="linear", C=1.0)),
        ])
    if kind == "xgb":
        if not HAS_XGB:
            raise ImportError("未安装 xgboost，请先 `pip install xgboost`")
        return XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"未知回归器类型: {kind}")


def evaluate_regressor(model: RegressorMixin, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    try:
        rmse = mean_squared_error(y, pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run_single_regression(
    name: str,
    reg_kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    has_val: bool = True,
) -> EvalResult:
    base_reg = make_regressor(reg_kind, random_state)

    if has_val:
        # 有验证集：在训练集上训练，用验证集选择超参数，然后用训练+验证集重新训练
        model = clone(base_reg)
        model.fit(X_train, y_train)
        val_metrics = evaluate_regressor(model, X_val, y_val)

        model_full = clone(base_reg)
        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)
        model_full.fit(X_train_full, y_train_full)
        test_pred = model_full.predict(X_test)
        test_metrics = evaluate_regressor(model_full, X_test, y_test)

        print(f"\n=== {name} ===")
        print(f"验证集: MAE={val_metrics['mae']:.4f} RMSE={val_metrics['rmse']:.4f} R²={val_metrics['r2']:.4f}")
        print(f"测试集: MAE={test_metrics['mae']:.4f} RMSE={test_metrics['rmse']:.4f} R²={test_metrics['r2']:.4f}")
    else:
        # 没有验证集：直接在训练集上训练，然后在测试集上评估
        model = clone(base_reg)
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        test_metrics = evaluate_regressor(model, X_test, y_test)
        # 没有验证集时，val_metrics使用测试集的结果（仅用于记录）
        val_metrics = test_metrics.copy()

        print(f"\n=== {name} ===")
        print(f"训练集大小: {len(y_train)} 样本")
        print(f"测试集: MAE={test_metrics['mae']:.4f} RMSE={test_metrics['rmse']:.4f} R²={test_metrics['r2']:.4f}")

    return EvalResult(
        name=name,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_true=y_test,
        test_pred=test_pred,
    )


def main():
    parser = argparse.ArgumentParser(description="MG2Act 回归基线：RF / SVR / XGBoost")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--val_csv", type=Path, required=False, help="验证集CSV（可选，如果不提供则在训练集上训练）")
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--morgan_radius", type=int, default=3)
    parser.add_argument("--morgan_bits", type=int, default=2048)
    parser.add_argument("--out_dir", type=Path, default=Path("baseline_regression_results"))
    args = parser.parse_args()

    set_seed(args.random_state)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train = load_split(args.train_csv)
    df_test = load_split(args.test_csv)

    y_train = df_train["Score"].astype(float).values
    y_test = df_test["Score"].astype(float).values

    # 验证集可选
    if args.val_csv and args.val_csv.exists():
        df_val = load_split(args.val_csv)
        y_val = df_val["Score"].astype(float).values
        has_val = True
    else:
        # 如果没有验证集，使用训练集作为验证
        df_val = df_train.copy()
        y_val = y_train.copy()
        has_val = False

    experiments: List[EvalResult] = []
    for mol_feat in ["morgan", "maccs"]:
        kwargs = {}
        if mol_feat == "morgan":
            kwargs["radius"] = args.morgan_radius
            kwargs["n_bits"] = args.morgan_bits

        tr_mol, tr_prot = build_feature_components(df_train, mol_feat, **kwargs)
        if has_val:
            va_mol, va_prot = build_feature_components(df_val, mol_feat, **kwargs)
        else:
            # 如果没有验证集，使用训练集特征
            va_mol, va_prot = tr_mol, tr_prot
        te_mol, te_prot = build_feature_components(df_test, mol_feat, **kwargs)

        # 分子特征固定投影到64维（Morgan:2048->64, MACCS:166->64）
        mol_input_dim = tr_mol.shape[1]
        tr_mol_proj = apply_linear_projection(tr_mol, mol_input_dim, 64, random_state=args.random_state)
        va_mol_proj = apply_linear_projection(va_mol, mol_input_dim, 64, random_state=args.random_state)
        te_mol_proj = apply_linear_projection(te_mol, mol_input_dim, 64, random_state=args.random_state)

        # 蛋白特征（E3+Target ACC）投影到64维
        prot_input_dim = tr_prot.shape[1]  # ~2140
        tr_prot_proj = apply_linear_projection(tr_prot, prot_input_dim, 64, random_state=args.random_state + 1)
        va_prot_proj = apply_linear_projection(va_prot, prot_input_dim, 64, random_state=args.random_state + 1)
        te_prot_proj = apply_linear_projection(te_prot, prot_input_dim, 64, random_state=args.random_state + 1)

        X_train = np.concatenate([tr_mol_proj, tr_prot_proj], axis=1)
        X_val = np.concatenate([va_mol_proj, va_prot_proj], axis=1)
        X_test = np.concatenate([te_mol_proj, te_prot_proj], axis=1)

        for reg_kind, reg_name in [("rf", "RandomForestReg"),
                                   ("svr", "SVR"),
                                   ("xgb", "XGBoost") if HAS_XGB else ("svr", None)]:
            if reg_name is None:
                continue
            exp_name = f"{reg_name}_{mol_feat}"
            result = run_single_regression(
                name=exp_name,
                reg_kind=reg_kind,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                random_state=args.random_state,
                has_val=has_val,
            )
            experiments.append(result)

    results_json = {
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv) if args.val_csv else None,
        "test_csv": str(args.test_csv),
        "has_validation": has_val,
        "random_state": args.random_state,
        "morgan_radius": args.morgan_radius,
        "morgan_bits": args.morgan_bits,
        "has_xgboost": HAS_XGB,
        "results": [
            {
                "name": r.name,
                "val_metrics": r.val_metrics,
                "test_metrics": r.test_metrics,
            }
            for r in experiments
        ],
    }

    out_path = out_dir / "regression_baselines.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 回归基线结果已保存到: {out_path}")

    # 保存测试集预测结果到 CSV
    if experiments:
        csv_data = {"true_value": y_test}
        for result in experiments:
            if result.test_true is not None and result.test_pred is not None:
                csv_data[f"{result.name}_predicted"] = result.test_pred

        df_predictions = pd.DataFrame(csv_data)
        csv_path = out_dir / "test_predictions.csv"
        df_predictions.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✅ 测试集预测结果已保存到: {csv_path}")

    if not HAS_XGB:
        print("⚠️ 未安装 xgboost，跳过 XGBoost 回归器。可通过 `pip install xgboost` 后重新运行。")


if __name__ == "__main__":
    main()

"""
# 使用验证集的情况：
python MG2Actgithub_v1/basline/baseline_regression.py \
    --train_csv MG_data/splits_output_e3p/train.csv \
    --val_csv MG_data/splits_output_e3p/val.csv \
    --test_csv MG_data/splits_output_e3p/test.csv \
    --random_state 42 \
    --morgan_radius 3 \
    --morgan_bits 2048 \
    --out_dir baseline_regression_1209

python MG2Actgithub_v1/basline/baseline_regression.py \
    --train_csv MG_data/splits_output_e3p30_x/train.csv \
    --val_csv MG_data/splits_output_e3p30_x/val.csv \
    --test_csv MG_data/splits_output_e3p30_x/test.csv \
    --random_state 42 \
    --morgan_radius 3 \
    --morgan_bits 2048 \
    --out_dir baseline_regression_1209_e3p30_x

python MG2Actgithub_v1/basline/baseline_regression.py \
    --train_csv MG_data/splits_output_e3p_fine/train.csv \
    --val_csv MG_data/splits_output_e3p_fine/val.csv \
    --test_csv MG_data/splits_output_e3p_fine/test.csv \
    --random_state 42 \
    --morgan_radius 3 \
    --morgan_bits 2048 \
    --out_dir baseline_regression_results_e3p_fine

# 只有训练和测试集的情况（跳过--val_csv参数）：
python MG2Actgithub_v1/basline/baseline_regression.py \
    --train_csv MG_data/splits_train_test_e3p/train.csv \
    --test_csv MG_data/splits_train_test_e3p/test.csv \
    --random_state 42 \
    --morgan_radius 3 \
    --morgan_bits 2048 \
    --out_dir baseline_regression_results_no_val_e3p

"""