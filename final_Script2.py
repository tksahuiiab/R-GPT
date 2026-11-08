import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Dependency Check ---
try:
    from ifeatpro.features import get_feature
except ImportError:
    print("ERROR: The 'ifeatpro' package is not installed. Please follow the setup instructions in README.md", file=sys.stderr)
    sys.exit(1)

try:
    import cnn_predictor
except ImportError as exc:
    print(f"ERROR importing 'cnn_predictor': {exc}", file=sys.stderr)
    print("If the message mentions NumPy, ensure 'onnxruntime' is installed and NumPy is compatible.", file=sys.stderr)
    print("Install with: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)

# --- CONFIGURATION ---
MODEL1_PATH = 'xgboost_model_cksaap1.pkl'
MODEL2_PATH = 'rf_model.pkl'
MODEL3_PATH = 'xgboost_model.pkl'
LABEL_ENCODER_PATH = 'label_encoder.joblib'

MODEL4_DIR = Path(__file__).resolve().parent
MODEL4_ARTIFACTS = [
    MODEL4_DIR / 'protein_cnn_final.onnx',
    MODEL4_DIR / 'protein_cnn_tokenizer.json',
]

MODEL1_POSITIVE_LABEL = 'R-gene'
MODEL2_POSITIVE_LABEL = 'classified'

def check_dependencies():
    files_to_check = [MODEL1_PATH, MODEL2_PATH, MODEL3_PATH, LABEL_ENCODER_PATH] + MODEL4_ARTIFACTS
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    if missing_files:
        print("ERROR: Missing files:", file=sys.stderr)
        for f in missing_files:
            print(f" - {f}", file=sys.stderr)
        return False
    return True

def add_header_to_feature_file(file_path):
    df = pd.read_csv(file_path, header=None)
    num_features = df.shape[1] - 1
    header = ['Seq_ID'] + [f'f{i+1}' for i in range(num_features)]
    df.columns = header
    df.to_csv(file_path, index=False)
    print(f"Added header to {file_path}")

def generate_and_process_features(fasta_file, output_dir):
    features_to_extract = ["cksaap", "apaac"]
    output_files = {}
    for feat in features_to_extract:
        print(f"Running {feat.upper()} feature extraction...")
        try:
            get_feature(fasta_file, feat, output_dir)
            generated_file = os.path.join(output_dir, f"{feat}.csv")
            if os.path.exists(generated_file):
                print(f"Successfully created '{generated_file}'")
                add_header_to_feature_file(generated_file)
                output_files[feat] = generated_file
            else:
                print(f"ERROR: iFeature-Pro did not generate the file for {feat.upper()}.", file=sys.stderr)
                return None, None
        except Exception as e:
            print(f"ERROR during {feat.upper()} feature extraction: {e}", file=sys.stderr)
            return None, None
    return output_files.get("cksaap"), output_files.get("apaac")

def load_model_and_data(model_path, data_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"ERROR loading model from '{model_path}': {e}", file=sys.stderr)
        return None, None
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"ERROR loading data from '{data_path}': {e}", file=sys.stderr)
        return None, None
    return model, data

def main(fasta_file, output_dir):
    if not Path(fasta_file).is_file():
        print(f"ERROR: Input FASTA file not found at '{fasta_file}'", file=sys.stderr)
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Step 0: Feature Extraction ===
    print("--- Step 0: Generating and Processing Features ---")
    cksaap_file, apaac_file = generate_and_process_features(fasta_file, output_dir)
    if not cksaap_file or not apaac_file:
        sys.exit(1)

    # === Step 1: R-gene Prediction ===
    print(f"\n--- Step 1: Predicting R-genes using '{MODEL1_PATH}' ---")
    model1, cksaap_data = load_model_and_data(MODEL1_PATH, cksaap_file)
    if model1 is None or cksaap_data is None: sys.exit(1)

    X_cksaap = cksaap_data.drop(columns=['Seq_ID'])
    predictions1 = model1.predict(X_cksaap)
    proba1 = model1.predict_proba(X_cksaap)[:, 1]
    cksaap_data['prediction_model1'] = predictions1
    r_gene_ids = cksaap_data[cksaap_data['prediction_model1'] == 1]['Seq_ID'].tolist()
    print(f"Found {len(r_gene_ids)} potential R-genes out of {len(cksaap_data)} total sequences.")

    model1_output = pd.DataFrame({
        'Seq_ID': cksaap_data['Seq_ID'],
        'Predicted_Label': predictions1,
        'Probability_R_gene': proba1
    })
    model1_output.to_csv(Path(output_dir) / 'model1_predictions.csv', index=False)

    '''if not r_gene_ids:
        print("Warning: No sequences classified as R-gene.")
        sys.exit(0)'''

    # === Step 2: Classified vs Unclassified ===
    print(f"\n--- Step 2: Classifying R-genes using '{MODEL2_PATH}' ---")
    model2, apaac_data = load_model_and_data(MODEL2_PATH, apaac_file)
    if model2 is None or apaac_data is None: sys.exit(1)

    apaac_data_filtered = apaac_data[apaac_data['Seq_ID'].isin(r_gene_ids)].copy()
    '''if apaac_data_filtered.empty:
        print("Data inconsistency: R-gene IDs not found in APAAC file.", file=sys.stderr)
        sys.exit(1)'''

    X_apaac_filtered = apaac_data_filtered.drop(columns=['Seq_ID'])
    predictions2 = model2.predict(X_apaac_filtered)
    proba2 = model2.predict_proba(X_apaac_filtered)[:, 1]
    apaac_data_filtered['prediction_model2'] = predictions2
    classified_ids = apaac_data_filtered[apaac_data_filtered['prediction_model2'] == 1]['Seq_ID'].tolist()
    print(f"Found {len(classified_ids)} potential 'classified' R-genes.")

    model2_output = pd.DataFrame({
        'Seq_ID': apaac_data_filtered['Seq_ID'],
        'Predicted_Label': predictions2,
        'Probability_Classified': proba2
    })
    model2_output.to_csv(Path(output_dir) / 'model2_predictions.csv', index=False)

    '''if not classified_ids:
        print("Warning: No R-genes classified as 'classified'.")
        sys.exit(0)'''

    # === Step 3: Multiclass Classification ===
    print(f"\n--- Step 3: Final multiclass prediction using '{MODEL3_PATH}' ---")
    model3, _ = load_model_and_data(MODEL3_PATH, cksaap_file)
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except FileNotFoundError:
        print(f"ERROR: Label encoder file not found at '{LABEL_ENCODER_PATH}'.", file=sys.stderr)
        sys.exit(1)

    if model3 is None: sys.exit(1)

    cksaap_data_final = cksaap_data[cksaap_data['Seq_ID'].isin(classified_ids)].copy()
    if cksaap_data_final.empty:
        print("Data inconsistency: Classified R-gene IDs not found in CKSAAP file.", file=sys.stderr)
        sys.exit(1)

    X_cksaap_final = cksaap_data_final.drop(columns=['Seq_ID', 'prediction_model1'])
    predictions3_encoded = model3.predict(X_cksaap_final)
    proba3 = model3.predict_proba(X_cksaap_final)
    decoded_labels = label_encoder.inverse_transform(predictions3_encoded)

    model3_output = pd.DataFrame({
        'Seq_ID': cksaap_data_final['Seq_ID'],
        'Predicted_Class': decoded_labels
    })
    for i, class_name in enumerate(label_encoder.classes_):
        model3_output[f'Prob_{class_name}'] = proba3[:, i]

    model3_output.to_csv(Path(output_dir) / 'model3_predictions.csv', index=False)

    # === Step 4: Pathogen-Type Prediction (CNN) ===
    print(f"\n--- Step 4: Predicting pathogen types using CNN ---")
    try:
        model4_output = cnn_predictor.predict_pathogens(
            fasta_file=fasta_file,
            r_gene_ids=r_gene_ids,
            model_dir=MODEL4_DIR,
        )
        model4_output.to_csv(Path(output_dir) / 'model4_predictions.csv', index=False)
        print(f"Pathogen predictions saved for {len(model4_output)} R-gene sequences.")
    except Exception as e:
        print(f"ERROR during CNN pathogen prediction: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Pipeline Complete ---")
    print(f"Final predictions saved to:'{output_dir}/model3_predictions.csv'")
    print(f"Pathogen predictions saved to:'{output_dir}/model4_predictions.csv'")
    print("\nFinal Results:")
    '''print(model3_output.to_string())'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A pipeline for sequential classification of protein sequences using four models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("fasta_file", type=str, help="Path to input FASTA file.")
    parser.add_argument("output_dir", type=str, help="Output directory for prediction results.")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    main(args.fasta_file, args.output_dir)
