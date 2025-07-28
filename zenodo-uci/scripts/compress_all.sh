#!/bin/bash
# This script compresses each subdirectory inside the provided complete/ folder.
# Only the .zip files inside each subdirectory are included.
# The resulting archive (named after the subdirectory) is created in the current working directory.
#
# Usage:
#   chmod +x compress_complete_subdirs.sh
#   ./compress_complete_subdirs.sh /path/to/complete/
#
# Example:
#   ./compress_complete_subdirs.sh /opt/jupyterhub/fast/shared_dir/UCI-Benchmark/complete

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <complete_directory>"
    exit 1
fi

COMPLETE_DIR="$1"

if [ ! -d "$COMPLETE_DIR" ]; then
    echo "Error: Directory $COMPLETE_DIR does not exist."
    exit 1
fi

for dir in "$COMPLETE_DIR"/*/; do
    # Get the subdirectory name.
    dir_name=$(basename "$dir")
    archive="${dir_name}.zip"
    echo "Compressing .zip files from ${dir} into ${archive}"
    zip -j "$archive" "${dir}"*.zip
done

echo "Compression complete. Archives are located in $(pwd)."
