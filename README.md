# R-GPT: R-Gene Prediction Tool

R-GPT is a four-stage machine learning pipeline for predicting plant resistance (R) genes from protein sequences and predicting their pathogen type.

## Pipeline overview

1. **Stage 1 — R-gene detection:** Binary classification of protein sequences as R-gene vs non-R-gene.
2. **Stage 2 — Classified vs unclassified:** Determines whether a predicted R-gene belongs to a known class.
3. **Stage 3 — R-gene class prediction:** Predicts one of seven R-gene classes.
4. **Stage 4 — Pathogen type prediction:** CNN-based prediction of the pathogen type (Fungus, Bacteria, Virus, Other).

## Output classes

- `NBS-LRR-CC`
- `NBS-LRR-TIR`
- `LRR-TrD`
- `LRR-TrD-Kinase`
- `Kinase-Kinase`
- `Lysin`
- `TrD-CC`
- `Unclassified`
- `Non-R`

## Installation

Create and activate the conda environment:

```bash
conda env create -f gene_pipeline_env.yml
conda activate gene_pipeline_env
```

## Usage

Run the full pipeline on a FASTA file:

```bash
python final_Script2.py input.fasta output_dir/
```

### Example

```bash
python final_Script2.py example/protein.fasta results/
```

## Output files

After running, `output_dir/` will contain:

- `model1_predictions.csv` — Stage 1 R-gene detection results
- `model2_predictions.csv` — Stage 2 classified vs unclassified results
- `model3_predictions.csv` — Stage 3 R-gene class predictions
- `model4_predictions.csv` — Stage 4 pathogen type predictions

## Repository structure

```
RGPT/
├── README.md
├── gene_pipeline_env.yml       # Conda environment
├── final_Script2.py            # Main pipeline script
├── cnn_predictor.py            # CNN pathogen predictor (Stage 4)
├── Encoder.py                  # Helper script
├── *.pkl / *.joblib / *.onnx   # Trained model files
├── example/
│   └── protein.fasta           # Example input
├── data/
│   ├── training/               # Training data
│   │   ├── training_pool.fasta # 2,825 R-gene sequences
│   │   └── negatives.fasta     # 2,825 non-R sequences
│   └── independent_test/       # Independent data
│       ├── final_independent_set.fasta
│       └── final_independent_labels.csv
└── docs/
    └── RGPT manual.pdf         # User manual
```

## Model files

| File | Description |
|---|---|
| `xgboost_model_cksaap1.pkl` | Stage 1: R-gene detection model |
| `rf_model.pkl` | Stage 2: Classified vs unclassified model |
| `xgboost_model.pkl` | Stage 3: Multiclass R-gene classifier |
| `label_encoder.joblib` | Class label encoder |
| `protein_cnn_final.onnx` | Stage 4: CNN pathogen type model |
| `protein_cnn_tokenizer.json` | CNN tokenizer |
| `protein_cnn_config.json` | CNN configuration |

## Datasets

- **Training set:** `data/training/training_pool.fasta` (2,825 R-genes) + `data/training/negatives.fasta` (2,825 non-R genes).
- **Independent test set:** `data/independent_test/final_independent_set.fasta` (286 R-genes + 40 non-R genes), with labels in `data/independent_test/final_independent_labels.csv`. This set was used for the RGPT vs PRGminer comparison.
