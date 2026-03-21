```markdown
# MG2Act: A Deep Learning Framework for PROTAC Degradation Activity Prediction
This resource library is presented together with the paper "MG2Act: MolGlueDB-Driven Quantitative AI Scoring of CRBN-Mediated Molecular Glue Degradation Efficacy" published by Zhuangzhiyao, Tengdan,XXX et al. in XXXX.

This repository contains the implementation of **MG2Act**, a deep learning framework for predicting Molecular Glue degradation activity by integrating:

- Protein sequence encoding (ESM-C)
- Molecular graph representation (GNN)
- Functional-group-aware attention fusion

---

## 📁 Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Model Weights](#2-model-weights)
- [3. Data Format](#3-data-format)
- [4. Training](#4-training)
- [5. Inference](#5-inference)
- [6. Notes](#6-notes)

---

## 1. Environment Setup

We recommend Python 3.10+.

Install dependencies according to your CUDA/PyTorch environment:

```bash
conda env create -f env.yaml
conda activate MG2Act

pip install torch pandas numpy tqdm scikit-learn
pip install rdkit
pip install torch-geometric
pip install esm
```

> For `torch` and `torch-geometric`, please use versions compatible with your local CUDA setup.

---

## 2. Model Weights

### 2.1 ESM-C Weights (Required)

Download ESM-C weights from:

https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12/tree/main/data

Then place the downloaded files into the `data/weight/` directory.

### 2.2 MG2Act Weights

Download pretrained MG2Act weights via:

```bash
git clone https://huggingface.co/944809681z/MG2Act
```

Use the downloaded model folder as `--model_dir` during inference.

---

## 3. Data Format

### 3.1 Training Data

Training expects a folder as `MG_data` containing:

- `train.csv`
- `val.csv`
- `test.csv`

Required columns:

- `E3_seq`
- `Target_seq`
- `Molecule_SMILES`
- `Score`

Recommended additional column:

- `PrimaryTarget`

### 3.2 Inference Data

Inference CSV must contain:

- `E3_seq`
- `Target_seq`
- `Molecule_SMILES`

---

## 4. Training

Run from the parent directory using module mode:
```bash
cd MG2Act
```
```bash
python -m MG2Actgithub_v1.train \
  --folder MG_data \
  --epochs 200 \
  --batch_size 8 \
  --lr 1e-4 \
  --device cuda:0 \
  --out model_dir  \
  --embed_dim 64 \
  --attn_heads 4 \
  --decoder_layers 2 \
  --mlp_hidden "128,64" \
  --dropout 0.1 \
  --proj_method conv \
  --gnn_type gat \
  --gnn_layers 3 \
  --gnn_hidden_dim 128 \
  --use_cosine \
  --early_stop_patience 30 \
  --fusion_method attention \
  --seed 42
```


Training outputs include the best checkpoint `mg2act_best.pt` and result metadata.

---

## 5. Inference

```bash
python -m MG2Actgithub_v1copy.predict_infer \
  --input /path/to/infer.csv \
  --model_dir /path/to/model_dir \
  --output /path/to/predictions.csv \
  --device cuda:0
```

The output CSV will include:

- `Predicted_Score`

---

## 6. Notes

- Use module execution (`python -m ...`) because the project uses relative imports.
- Ensure ESM-C weights are correctly placed in `data/weight/` before running.
- RDKit and PyG installation may vary by system/CUDA; verify compatibility first.

---

## Citation

If you use this repository in your research, please cite your MG2Act paper/project accordingly.
```
