"""
cnn_predictor.py
----------------
Self-contained pathogen-type predictor for R-gene sequences.

Loads the saved ONNX model and tokenizer from the same directory and returns
multi-label pathogen predictions for a list of sequence IDs.

Used by final_Script2.py as Model 4 of the RGPT webserver.

Deployment requirement:
    pip install onnxruntime
"""

import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError(
        "onnxruntime is required for Model 4 (CNN pathogen prediction). "
        "Install it with: pip install onnxruntime"
    ) from exc


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<pad>", "<unk>"]
ALL_TOKENS = SPECIAL_TOKENS + AMINO_ACIDS


class AminoAcidTokenizer:
    """Character-level tokenizer for protein sequences."""

    def __init__(self):
        self.token_to_id = {token: idx for idx, token in enumerate(ALL_TOKENS)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.unk_token_id = self.token_to_id["<unk>"]
        self.vocab_size = len(ALL_TOKENS)

    def encode(self, sequence: str) -> List[int]:
        return [
            self.token_to_id.get(aa.upper(), self.unk_token_id)
            for aa in sequence
        ]


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------
def short_id(seq_id: str) -> str:
    """Return the first whitespace-delimited token of a sequence ID."""
    return re.split(r"\s+", str(seq_id).strip())[0]


def parse_fasta(fasta_path: str) -> Tuple[List[str], List[str]]:
    """Parse a FASTA file and return (short_ids, sequences)."""
    ids = []
    sequences = []
    current_id = None
    current_seq = []
    seen = set()

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sid = short_id(current_id)
                    if sid not in seen:
                        seen.add(sid)
                        ids.append(sid)
                        sequences.append("".join(current_seq))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            sid = short_id(current_id)
            if sid not in seen:
                seen.add(sid)
                ids.append(sid)
                sequences.append("".join(current_seq))

    return ids, sequences


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_onnx_session(model_dir: Path):
    """Load the ONNX model and return an inference session."""
    onnx_path = model_dir / "protein_cnn_final.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")

    providers = ort.get_available_providers()
    # Prefer CPUExecutionProvider for deployment consistency.
    if "CPUExecutionProvider" in providers:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(str(onnx_path), providers=providers)


def load_tokenizer(model_dir: Path) -> AminoAcidTokenizer:
    """Load tokenizer vocabulary from JSON."""
    tokenizer_path = model_dir / "protein_cnn_tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer config: {tokenizer_path}")

    with open(tokenizer_path, "r") as fh:
        tokenizer_config = json.load(fh)

    tokenizer = AminoAcidTokenizer()
    tokenizer.token_to_id = {
        k: int(v) for k, v in tokenizer_config["token_to_id"].items()
    }
    tokenizer.id_to_token = {
        int(v): k for k, v in tokenizer_config["token_to_id"].items()
    }
    tokenizer.pad_token_id = int(tokenizer_config["pad_token_id"])
    tokenizer.unk_token_id = int(tokenizer_config["unk_token_id"])
    tokenizer.vocab_size = len(tokenizer.token_to_id)
    return tokenizer


def collate_sequences(sequences: List[str], tokenizer: AminoAcidTokenizer) -> np.ndarray:
    """Encode and pad a list of sequences to the same length."""
    encoded = [tokenizer.encode(seq) for seq in sequences]
    max_len = max(len(seq) for seq in encoded)
    padded = np.full((len(encoded), max_len), tokenizer.pad_token_id, dtype=np.int64)
    for i, seq in enumerate(encoded):
        padded[i, : len(seq)] = seq
    return padded


def predict_pathogens(
    fasta_file: str,
    r_gene_ids: List[str],
    model_dir: Path,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Predict pathogen types for the given R-gene IDs using the ONNX model.

    Returns a DataFrame with one row per R-gene sequence.
    """
    labels = ["Fungus", "Bacteria", "Virus", "Other"]

    if not r_gene_ids:
        return pd.DataFrame(
            columns=[
                "Seq_ID",
                "Prob_Fungus",
                "Prob_Bacteria",
                "Prob_Virus",
                "Prob_Other",
                "Pred_Fungus",
                "Pred_Bacteria",
                "Pred_Virus",
                "Pred_Other",
                "Predicted_Pathogens",
            ]
        )

    session = load_onnx_session(model_dir)
    tokenizer = load_tokenizer(model_dir)

    all_ids, all_seqs = parse_fasta(fasta_file)
    id_to_seq = dict(zip(all_ids, all_seqs))

    selected_ids = []
    selected_seqs = []
    missing = []
    for sid in r_gene_ids:
        if sid in id_to_seq:
            selected_ids.append(sid)
            selected_seqs.append(id_to_seq[sid])
        else:
            missing.append(sid)

    if missing:
        print(
            f"Warning: {len(missing)} R-gene IDs not found in FASTA; skipping them.",
            file=__import__("sys").stderr,
        )

    all_probas = []
    input_name = session.get_inputs()[0].name
    for i in range(0, len(selected_seqs), batch_size):
        batch_seqs = selected_seqs[i : i + batch_size]
        x = collate_sequences(batch_seqs, tokenizer)
        logits = session.run(None, {input_name: x})[0]
        proba = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        all_probas.append(proba)

    probabilities = np.vstack(all_probas)
    predictions = (probabilities >= threshold).astype(int)

    out = pd.DataFrame({"Seq_ID": selected_ids})
    for j, label in enumerate(labels):
        out[f"Prob_{label}"] = probabilities[:, j]
    for j, label in enumerate(labels):
        out[f"Pred_{label}"] = predictions[:, j]

    predicted_pathogens = []
    for row in predictions:
        labels_above = [labels[j] for j, val in enumerate(row) if val == 1]
        predicted_pathogens.append(
            ", ".join(labels_above) if labels_above else "None"
        )
    out["Predicted_Pathogens"] = predicted_pathogens

    return out
