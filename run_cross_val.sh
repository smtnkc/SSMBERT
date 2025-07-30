#!/bin/bash

# Arrays of parameters
datasets=("heterounique" "heterogrouped" "case1" "case2")
targets=("entry" "read" "write" "exit" "interaction" "communication" "process")
models=("bert" "se-bert")

# Total number of combinations for progress tracking
total=$((${#datasets[@]} * ${#targets[@]} * ${#models[@]}))
current=0

# Function to show progress
show_progress() {
    local current=$1
    local total=$2
    local percentage=$((current * 100 / total))
    echo "Progress: $current/$total ($percentage%)"
}

# Main loop
for model in "${models[@]}"; do
    echo "============================================="
    echo "Starting experiments with model: $model"
    echo "============================================="
    
    for dataset in "${datasets[@]}"; do
        echo "---------------------------------------------"
        echo "Processing dataset: $dataset"
        echo "---------------------------------------------"
        
        for target in "${targets[@]}"; do
            ((current++))
            echo "Running combination $current of $total"
            echo "Model: $model"
            echo "Dataset: $dataset"
            echo "Target: $target"
            echo "---------------------------------------------"
            
            python cross_val.py --dataset "$dataset" --target "$target" --model "$model"
            
            show_progress $current $total
            echo -e "\n"
        done
    done
done

echo "All combinations completed!" 