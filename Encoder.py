import pandas as pd
import os
from ifeatpro.features import get_feature
fasta_file = "proteins.fasta"
output_folder = "trial/"
features = ["apaac","cksaap"]

#generate the features
for feat in features:
    get_feature(fasta_file, feat, output_folder)

def add_header_to_feature_file(file_path):
    df = pd.read_csv(file_path, header=None)
    
    # First column is Seq_ID, rest are features
    num_features = df.shape[1] - 1
    header = ['Seq_ID'] + [f'f{i+1}' for i in range(num_features)]
    
    df.columns = header
    df.to_csv(file_path, index=False)

# Apply to both feature files
output_folder = "trial/"
feature_files = ["apaac.csv", "cksaap.csv"]

for file_name in feature_files:
    full_path = os.path.join(output_folder, file_name)
    add_header_to_feature_file(full_path)
