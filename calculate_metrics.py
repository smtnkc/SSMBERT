import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def extract_info_from_filename(filename):
    # Example: case1-shuffled-data-predby-bert-heterounique.csv
    parts = filename.replace('.csv', '').split('-')
    test_data = parts[0]  # case1
    model = parts[4]  # bert
    if parts[4] == 'se':
        model = 'se-' + parts[5]  # se-bert
        train_data = parts[6]  # hetero
    else:
        train_data = parts[5]  # hetero
    return model, train_data, test_data

def calculate_metrics(actuals, predictions):
    mse = round(float(np.mean((predictions - actuals)**2)), 4)
    mae = round(float(np.mean(np.abs(predictions - actuals))), 4)
    nmae = round(mae / np.mean(actuals) if np.mean(actuals) != 0 else float('inf'), 4)
    round_preds = np.round(predictions)
    exact_match_acc = round(accuracy_score(actuals, round_preds), 4)
    
    # Calculate MMRE and PRED(30)
    valid_indices = actuals != 0
    nonzero_perc = round(float(np.mean(valid_indices) * 100), 4)
    
    if np.any(valid_indices):
        mre = np.abs(predictions[valid_indices] - actuals[valid_indices]) / actuals[valid_indices]
        mmre = round(float(np.mean(mre)), 4)
        pred30 = round(float(np.mean(mre <= 0.30)), 4)
    else:
        mmre = -1
        pred30 = -1
        
    return {
        'mse': mse,
        'mae': mae,
        'nmae': nmae,
        'acc': exact_match_acc,
        'mmre': mmre,
        'pred30': pred30,
        'nonzero': nonzero_perc
    }

def main():
    # Create stats directory if it doesn't exist
    os.makedirs('stats', exist_ok=True)

    # Initialize DataFrames for each metric
    metrics_dfs = {
        'mse': [],
        'mae': [],
        'nmae': [],
        'acc': [],
        'mmre': [],
        'pred30': [],
        'nonzero': []
    }
    
    # Process each prediction file
    for filename in os.listdir('preds'):
        if not filename.endswith('.csv'):
            continue
            
        # Extract information from filename
        model, train_data, test_data = extract_info_from_filename(filename)
        
        # Read the prediction file
        df = pd.read_csv(f'preds/{filename}')
        
        # Features to evaluate in the desired order
        features = ['entry', 'read', 'write', 'exit', 'cfp', 'interaction', 'communication', 'process', 'microm']
        
        # Calculate metrics for each feature
        for feature in features:
            actuals = df[feature].values
            predictions = df[f'{feature}_pred'].values
            
            # Calculate all metrics
            metrics = calculate_metrics(actuals, predictions)
            
            # Add results to each metric DataFrame
            for metric_name, metric_value in metrics.items():
                metrics_dfs[metric_name].append({
                    'model': model,
                    'train_set': train_data,
                    'test_set': test_data,
                    'feature': feature,
                    'value': metric_value
                })
    
    # Convert to DataFrames and save to CSV
    for metric_name, rows in metrics_dfs.items():
        df = pd.DataFrame(rows)
        # Pivot the DataFrame to get features as columns
        df_pivot = df.pivot_table(
            index=['model', 'train_set', 'test_set'],
            columns='feature',
            values='value'
        ).reset_index()
        
        # Ensure columns are in the correct order
        column_order = ['model', 'train_set', 'test_set'] + features
        df_pivot = df_pivot[column_order]
        
        # Save to CSV
        df_pivot.to_csv(f'stats/{metric_name}.csv', index=False)

if __name__ == "__main__":
    main() 