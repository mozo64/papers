#!/bin/bash
# This script compresses all "model_tf" directories found under the given parent directory.
# The resulting ZIP archives are placed in the "models" subdirectory of the current working directory.
#
# Usage: ./compress_models.sh <parent_directory>
# Example: ./compress_models.sh ../ds
#
# Each archive is named using the relative path (with "/" replaced by "_") from the parent directory.
# Any leading dots in the filename are removed.
#
# Unzipping the archive will recreate the original "model_tf" directory.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

# Remove any trailing slash from the parent directory.
PARENT_DIR="${1%/}"

if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Directory $PARENT_DIR does not exist."
    exit 1
fi

# Set the output directory to a subfolder "models" in the current working directory.
OUTPUT_DIR="$(pwd)/models"
mkdir -p "$OUTPUT_DIR"

# Find all directories named "model_tf" under the parent directory.
find "$PARENT_DIR" -type d -name "model_tf" | while read -r model_dir; do
    # Compute the relative path from the parent directory.
    # Example: if model_dir is "../ds/multivariate/SelfRegulationSCP1/model_tf"
    # then relative path becomes "multivariate/SelfRegulationSCP1/model_tf".
    rel_path="${model_dir#$PARENT_DIR/}"
    
    # Create a flat filename by replacing "/" with "_" and appending ".zip".
    archive_file=$(echo "$rel_path" | tr '/' '_').zip
    
    # Remove any leading dots in the archive filename.
    archive_file=$(echo "$archive_file" | sed 's/^\.+//')
    
    archive_path="$OUTPUT_DIR/$archive_file"
    
    echo "Compressing $model_dir into $archive_path"
    
    (
        cd "$(dirname "$model_dir")" || exit
        zip -r "$archive_path" "$(basename "$model_dir")"
    )
done
