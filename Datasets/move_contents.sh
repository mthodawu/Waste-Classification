#!/bin/bash

# Base directory (the folder containing 'other' directories like 'plastic_and_other', 'metal_and_other')
BASE_DIR="."

# Loop through each directory that ends with '_and_other'
for MAIN_DIR in "$BASE_DIR"/*_and_other; do
    if [ -d "$MAIN_DIR/other" ]; then
        echo "Processing '$MAIN_DIR/other'..."

        # Loop through all subfolders in 'other'
        for SUBFOLDER in "$MAIN_DIR/other"/*; do
            if [ -d "$SUBFOLDER" ]; then
                echo "Moving contents of '$SUBFOLDER' to '$MAIN_DIR/other/'..."

                # Move the contents of the subfolder to the 'other' directory
                mv "$SUBFOLDER"/* "$MAIN_DIR/other/"

                # Remove the now empty subfolder
                rmdir "$SUBFOLDER"
            fi
        done

        echo "Completed processing '$MAIN_DIR/other'."
    else
        echo "'other' directory not found in '$MAIN_DIR'. Skipping..."
    fi
done

echo "All 'other' directories have been processed."
