import os
import pandas as pd

def process_file(file_path):
    print(f"Processing {file_path}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    try:
        # Calculate CFP (Control Flow Primitives) prediction
        df['cfp_pred'] = (
            df['entry_pred'] + 
            df['read_pred'] + 
            df['write_pred'] + 
            df['exit_pred']
        )
        
        # Calculate MicroM (Microservices Metrics) prediction
        df['microm_pred'] = (
            df['interaction_pred'] + 
            df['communication_pred'] + 
            df['process_pred']
        )
        
        # Reorder columns to put cfp_pred after exit_pred
        cols = list(df.columns)
        
        # Find the position after exit_pred
        exit_pos = cols.index('exit_pred')
        
        # Remove the new columns from their current positions
        cols.remove('cfp_pred')
        cols.remove('microm_pred')
        
        # Insert cfp_pred after exit_pred
        cols.insert(exit_pos + 1, 'cfp_pred')
        
        # Add microm_pred at the end
        cols.append('microm_pred')
        
        # Reorder the DataFrame
        df = df[cols]
        
        # Save the modified DataFrame back to the same file
        df.to_csv(file_path, index=False)
        print(f"Successfully processed {file_path}")
        
    except KeyError as e:
        print(f"Warning: Skipping {file_path} - Missing required column: {e}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Get all CSV files in the preds directory
    preds_dir = 'preds'
    if not os.path.exists(preds_dir):
        print(f"Error: {preds_dir} directory not found!")
        return

    csv_files = [f for f in os.listdir(preds_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {preds_dir} directory!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    for file_name in csv_files:
        file_path = os.path.join(preds_dir, file_name)
        process_file(file_path)

if __name__ == "__main__":
    main() 