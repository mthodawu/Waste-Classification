#!/bin/bash

# Source directory
SRC_DIR="Trashnet-resized"

# List of materials to exclude from 'other' directory
MATERIALS=("plastic" "metal" "cardboard" "paper" "trash")

# Loop through each material
for MATERIAL in "${MATERIALS[@]}"; do
    # Create new directory with the format '{material}_and_other'
    DEST_DIR="${MATERIAL}_and_other"
    mkdir -p "$DEST_DIR"

    # Copy the specific material folder to the new directory
    cp -r "$SRC_DIR/$MATERIAL" "$DEST_DIR/"

    # Create 'other' directory inside the new destination directory
    mkdir -p "$DEST_DIR/other"

    # Loop through all subdirectories in the source directory
    for SUBFOLDER in "$SRC_DIR"/*; do
        # Get the name of the subfolder
        FOLDER_NAME=$(basename "$SUBFOLDER")

        # Copy subfolder to 'other' directory, excluding the material
        if [[ "$FOLDER_NAME" != "$MATERIAL" ]]; then
            cp -r "$SUBFOLDER" "$DEST_DIR/other/"
        fi
    done

    echo "Copied folders to '$DEST_DIR' with '$MATERIAL' and the rest in 'other'."
done

echo "All folders have been processed."
