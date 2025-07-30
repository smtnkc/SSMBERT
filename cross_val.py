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
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from transformers import logging as transformers_logging
from transformers import get_linear_schedule_with_warmup

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
    description="Cross-validated BERT regression",
    formatter_class=default_help
)
parser.add_argument(
    '--dataset',
    type=str,
    default='hetero',
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
    choices=['bert', 'se-bert'],
    required=True,
    help='Model variant to use'
)

args = parser.parse_args()

dataset = args.dataset
target = args.target
in_model_name = args.model

if in_model_name == 'bert':
    base_model = 'bert-base-uncased'
elif in_model_name == 'se-bert':
    base_model = 'burakkececi/bert-software-engineering'
else:
    raise ValueError(f"Invalid model name: {in_model_name}")

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
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('preds', exist_ok=True)

# File paths
shuffled_csv    = f"data/{dataset}-shuffled-data.csv"
out_model_name  = f"{in_model_name}-{dataset}" # bert-hetero
output_csv      = f"preds/{dataset}-shuffled-data-predby-{out_model_name}.csv" # hetero-shuffled-data-predby-bert-hetero.csv
log_path        = f"logs/{dataset}-{target}-predby-{out_model_name}.csv" # Changed extension to .csv
out_model_path  = f"checkpoints/{out_model_name}-{target}.pt" # bert-hetero-entry.pt

# Initialize log DataFrame with train and validation loss
log_df = pd.DataFrame(columns=[
    'Fold', 'Epoch', 'Train_Loss', 'Val_Loss',
    'MSE', 'MAE', 'NMAE', 'MMRE', 'PRED30', 'ACC', 'NonZeroPerc'
])

# Load dataset - if predictions exist, use that version, otherwise use original
if os.path.exists(output_csv):
    df_with_predictions = pd.read_csv(output_csv)
elif os.path.exists(shuffled_csv):
    df_with_predictions = pd.read_csv(shuffled_csv)
else:
    raise FileNotFoundError(f"Input dataset '{shuffled_csv}' not found.")

# Split into 5 folds
splits = np.array_split(df_with_predictions, 5)

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
tokenizer = BertTokenizer.from_pretrained(base_model)

best_mse = float('inf')
best_info = {'fold': None, 'epoch': None}
all_fold_predictions = []

# Cross-validation
for fold in range(5):
    test_df  = splits[fold]
    train_df = pd.concat([splits[i] for i in range(5) if i != fold], ignore_index=True)

    train_ds = TextRegressionDataset(
        train_df['requirement_en'], train_df[target], tokenizer
    )
    test_ds  = TextRegressionDataset(
        test_df['requirement_en'],  test_df[target],  tokenizer
    )

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        pin_memory=True, num_workers=4
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=32,
        pin_memory=True, num_workers=4
    )

    # Initialize and load model
    model = BertForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        problem_type='regression'
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.MSELoss()
    
    # Scheduler for learning rate
    total_steps = len(train_loader) * 10  # 10 epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Track best epoch for this fold
    fold_best_mse = float('inf')
    fold_best_preds = None

    for epoch in range(1, 11):
        # Training
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device, non_blocking=True) for k,v in batch.items() if k!='labels'}
            labels = batch['labels'].to(device, non_blocking=True).unsqueeze(1)
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        preds = []
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(device, non_blocking=True) for k,v in batch.items() if k!='labels'}
                labels = batch['labels'].to(device, non_blocking=True).unsqueeze(1)
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()
                logits = model(**inputs).logits.squeeze(-1).cpu().numpy()
                preds.extend(logits.tolist())
        avg_val_loss = total_val_loss / len(test_loader)
        actuals = test_df[target].values
        preds_array = np.array(preds)
        
        # Calculate metrics
        mse_epoch = float(np.mean((preds_array - actuals)**2))
        mae_epoch = float(np.mean(np.abs(preds_array - actuals)))
        nmae_epoch = mae_epoch / np.mean(actuals)  # Normalized MAE = MAE / mean(y)
        round_preds = np.round(preds_array)
        exact_match_acc = accuracy_score(actuals, round_preds)
        
        # Calculate MMRE and PRED(30)
        # Filter out cases where actual value is zero
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
            'Fold': [fold+1], 'Epoch': [epoch],
            'Train_Loss': [avg_train_loss], 'Val_Loss': [avg_val_loss],
            'MSE': [mse_epoch], 'MAE': [mae_epoch], 'NMAE': [nmae_epoch],
            'MMRE': [mmre], 'PRED30': [pred30], 'ACC': [exact_match_acc],
            'NonZeroPerc': [nonzero_perc]
        })
        log_df = pd.concat([log_df, new_log], ignore_index=True)
        # Save the updated log DataFrame to CSV
        log_df.to_csv(log_path, index=False)

        # Print metrics header once
        if epoch == 1 and fold == 0:
            print("\n{:<5} {:<5} {:<12} {:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Fold", "Epoch", "Train_Loss", "Val_Loss",
                "MSE", "MAE", "NMAE", "MMRE", "PRED(30)", "ACC", "NonZero%"
            ))
            print("-" * 120)
        # Print row
        print("{:<5} {:<5} {:<12.4f} {:<12.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.1f}".format(
            fold+1, epoch, avg_train_loss, avg_val_loss,
            mse_epoch, mae_epoch, nmae_epoch, mmre, pred30,
            exact_match_acc, nonzero_perc
        ))

        # Save best checkpoint and predictions
        if mse_epoch < best_mse:
            best_mse = mse_epoch
            best_info = {'fold': fold+1, 'epoch': epoch}
            torch.save(model.state_dict(), out_model_path)
        
        # Track best predictions for this fold
        if mse_epoch < fold_best_mse:
            fold_best_mse = mse_epoch
            fold_best_preds = preds.copy()  # Store a copy of the predictions

    print("-" * 120)
    all_fold_predictions.append((test_df.index, fold_best_preds))

# Merge and round predictions
indices, preds = zip(*all_fold_predictions)
flat_idx = np.concatenate(indices)
flat_preds = np.concatenate(preds)
round_preds = np.round(flat_preds).astype(int)  # Convert to integers after rounding

# Attach predictions and write output
out_column = f"{target}_pred"
df_with_predictions[out_column] = pd.Series(round_preds, index=flat_idx, dtype=int)  # Ensure integer type in pandas
df_with_predictions.to_csv(output_csv, index=False)

print(f"Done for dataset '{dataset}' with target '{target}' using model '{in_model_name}'.")
print(f"Best MSE {best_mse:.4f} at fold {best_info['fold']}, epoch {best_info['epoch']}.")
print(f"Best model saved to {out_model_path}")
print(f"Logs saved to {log_path}.")