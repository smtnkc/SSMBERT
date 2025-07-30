#!/bin/bash

# Print what we're going to delete
echo "The following directories will be cleaned:"
echo "- checkpoints/"
echo "- logs/"
echo "- preds/"

# Ask for confirmation
read -p "Are you sure you want to proceed? (y/N) " -n 1 -r
echo # Move to a new line

if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Remove directories and their contents
    rm -rf checkpoints/
    rm -rf logs/
    rm -rf preds/
    
    echo "Clean up completed successfully!"
else
    echo "Operation cancelled."
fi