import os
import warnings
import logging

# Suppress all warnings and logging before other imports
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # Disable all TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'           # Only use GPU 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # Disable oneDNN optimizations
os.environ['TF_SILENCE_TENSORFLOW'] = '1'          # Silence TensorFlow messages
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'     # Set transformers logging to error only
os.environ['TOKENIZERS_PARALLELISM'] = 'false'     # Disable tokenizers parallelism warning

import random
import pandas as pd
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from transformers import logging as transformers_logging
import time

# Additional warning suppression after imports
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Disable TensorFlow logging completely
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

# Parse arguments
default_help = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(
    description="BERT regression prediction",
    formatter_class=default_help
)
parser.add_argument(
    '--dataset',
    type=str,
    default='case1',
    help='Dataset prefix (without extension)'
)
parser.add_argument(
    '--target',
    type=str,
    default='entry',
    help='Name of the target column'
)
parser.add_argument(
    '--model',
    type=str,
    choices=['bert-hetero', 'se-bert-hetero', 'bert-heterounique', 'se-bert-heterounique'],
    required=True,
    help='Model variant to use'
)

args = parser.parse_args()

dataset = args.dataset
target = args.target
in_model_name = args.model
in_model_path = f"checkpoints/{in_model_name}-{target}.pt"

# Check if checkpoint exists
if not os.path.isfile(in_model_path):
    raise FileNotFoundError(f"Checkpoint file '{in_model_path}' not found. Please provide a valid path.")

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA-enabled GPU is required to run this script.")
torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('preds', exist_ok=True)

# File paths
shuffled_csv = f"data/{dataset}-shuffled-data.csv" # case1_shuffled_data.csv
output_csv = f"preds/{dataset}-shuffled-data-predby-{in_model_name}.csv" # case1-shuffled-data-predby-bert-hetero.csv
log_path = f"logs/{dataset}-{target}-predby-{in_model_name}.csv" # Changed extension to csv

# Initialize log DataFrame
log_df = pd.DataFrame(columns=['MSE', 'MAE', 'NMAE', 'ACC', 'MMRE', 'PRED30', 'NonZeroPerc'])

# Load dataset - if predictions exist, use that version, otherwise use original
if os.path.exists(output_csv):
    df_with_predictions = pd.read_csv(output_csv)
elif os.path.exists(shuffled_csv):
    df_with_predictions = pd.read_csv(shuffled_csv)
else:
    raise FileNotFoundError(f"Input dataset '{shuffled_csv}' not found.")

class TextRegressionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=256):
        self.texts = texts.tolist()
        self.targets = targets.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item

# Prepare tokenizer
if in_model_name.startswith('bert'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif in_model_name.startswith('se-bert'):
    tokenizer = BertTokenizer.from_pretrained('burakkececi/bert-software-engineering')
else:
    raise ValueError(f"Invalid model name: {in_model_name}")

# Create dataset and dataloader
ds = TextRegressionDataset(df_with_predictions['requirement_en'], df_with_predictions[target], tokenizer)
dataloader = DataLoader(ds, batch_size=32, pin_memory=True, num_workers=4)

# Load model
if in_model_name.startswith('bert'):
    base_model = 'bert-base-uncased'
elif in_model_name.startswith('se-bert'):
    base_model = 'burakkececi/bert-software-engineering'
else:
    raise ValueError(f"Invalid model name: {in_model_name}")

model = BertForSequenceClassification.from_pretrained(
    base_model,
    num_labels=1,
    problem_type='regression'
).to(device)

# Load the saved state dict
model.load_state_dict(torch.load(in_model_path))

# Perform prediction
model.eval()
all_preds = []
time_start = time.time()
with torch.no_grad():
    for batch in dataloader:
        inputs = {k: v.to(device, non_blocking=True) for k,v in batch.items() if k!='labels'}
        logits = model(**inputs).logits.squeeze(-1).cpu().numpy()
        all_preds.extend(logits.tolist())
time_end = time.time()
print(f"Prediction completed in {time_end - time_start:.2f} seconds.")

preds_array = np.array(all_preds)
actuals = df_with_predictions[target].values

# Calculate metrics
mse = float(np.mean((preds_array - actuals)**2))
mae = float(np.mean(np.abs(preds_array - actuals)))
nmae = mae / np.mean(actuals)  # Normalized MAE = MAE / mean(y)
round_preds = np.round(preds_array)
exact_match_acc = accuracy_score(actuals, round_preds)

# Calculate MMRE and PRED(30)
valid_indices = actuals != 0
nonzero_perc = float(np.mean(valid_indices) * 100)  # Percentage of non-zero values
if np.any(valid_indices):
    mre = np.abs(preds_array[valid_indices] - actuals[valid_indices]) / actuals[valid_indices]
    mmre = float(np.mean(mre))
    pred30 = float(np.mean(mre <= 0.30))  # percentage of predictions with MRE <= 0.30
else:
    mmre = -1
    pred30 = -1

# Log metrics
new_log = pd.DataFrame({
    'MSE': [mse],
    'MAE': [mae],
    'NMAE': [nmae],
    'ACC': [exact_match_acc],
    'MMRE': [mmre],
    'PRED30': [pred30],
    'NonZeroPerc': [nonzero_perc]
})
log_df = pd.concat([log_df, new_log], ignore_index=True)
log_df.to_csv(log_path, index=False)

# Print metrics as table
print("\n{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "MSE", "MAE", "NMAE", "ACC", "MMRE", "PRED(30)", "NonZero%"
))
print("-" * 80)  # Separator line
print("{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.1f}".format(
    mse, mae, nmae, exact_match_acc, mmre, pred30, nonzero_perc
))
print("-" * 80)  # Separator line

# Convert predictions to integers and save
round_preds = np.round(preds_array).astype(int)
out_column = f"{target}_pred"
df_with_predictions[out_column] = round_preds
df_with_predictions.to_csv(output_csv, index=False)

print(f"\nDone for dataset '{dataset}' with target '{target}' using model '{in_model_name}'.")
print(f"Predictions saved to {output_csv}")
print(f"Logs saved to {log_path}.")