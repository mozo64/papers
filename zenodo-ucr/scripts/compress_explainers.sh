#!/bin/bash
# This script compresses specific pickle files from each series folder under
# ds/univariate and ds/multivariate directories.
#
# For each series folder (located in either univariate or multivariate), it creates:
# 1. A shap_values archive containing: svts.pickle and svtr.pickle
# 2. A lime_values archive containing: lvts.pickle and lvtr.pickle
# 3. An anchor_values archive containing: avts.pickle and avtr.pickle
# 4. A train_and_test archive containing: trainX.pickle, trainy.pickle, testX.pickle, testy.pickle
#
# The archives are placed in separate subdirectories in the current directory:
# shap, lime, anchor, train_test.
#
# The archive files are named in the format:
#   <dataset>_<series>_<archive_type>.zip
# where <dataset> is either 'multivariate' or 'univariate', and <archive_type> ends with:
#   shap_values.zip, lime_values.zip, anchor_values.zip, or train_and_test.zip.
#
# Usage: ./compress_files.sh <parent_directory>
# Example: ./compress_files.sh ../ds/

# Check that one argument (the parent directory) is provided.
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

PARENT_DIR="$1"

# Verify that the parent directory exists.
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Directory $PARENT_DIR does not exist."
    exit 1
fi

# Create output subdirectories in the current working directory.
mkdir -p shap lime anchor train_test

# Process both 'multivariate' and 'univariate' datasets.
for dataset in multivariate univariate; do
    dataset_dir="$PARENT_DIR/$dataset"
    if [ ! -d "$dataset_dir" ]; then
        echo "Directory not found: $dataset_dir. Skipping..."
        continue
    fi

    # Iterate over each series folder within the dataset directory.
    for series_dir in "$dataset_dir"/*; do
        if [ -d "$series_dir" ]; then
            series_name=$(basename "$series_dir")
            echo "Processing series: $dataset/$series_name"

            # Compress shap values (svts.pickle and svtr.pickle)
            if [ -f "$series_dir/svts.pickle" ] && [ -f "$series_dir/svtr.pickle" ]; then
                archive_name="${dataset}_${series_name}_shap_values.zip"
                echo "  Creating shap archive: $archive_name"
                zip -j "shap/$archive_name" "$series_dir/svts.pickle" "$series_dir/svtr.pickle"
            else
                echo "  Shap files missing for $series_name. Skipping shap archive."
            fi

            # Compress lime values (lvts.pickle and lvtr.pickle)
            if [ -f "$series_dir/lvts.pickle" ] && [ -f "$series_dir/lvtr.pickle" ]; then
                archive_name="${dataset}_${series_name}_lime_values.zip"
                echo "  Creating lime archive: $archive_name"
                zip -j "lime/$archive_name" "$series_dir/lvts.pickle" "$series_dir/lvtr.pickle"
            else
                echo "  Lime files missing for $series_name. Skipping lime archive."
            fi

            # Compress anchor values (avts.pickle and avtr.pickle)
            if [ -f "$series_dir/avts.pickle" ] && [ -f "$series_dir/avtr.pickle" ]; then
                archive_name="${dataset}_${series_name}_anchor_values.zip"
                echo "  Creating anchor archive: $archive_name"
                zip -j "anchor/$archive_name" "$series_dir/avts.pickle" "$series_dir/avtr.pickle"
            else
                echo "  Anchor files missing for $series_name. Skipping anchor archive."
            fi

            # Compress train and test files (trainX.pickle, trainy.pickle, testX.pickle, testy.pickle)
            if [ -f "$series_dir/trainX.pickle" ] && [ -f "$series_dir/trainy.pickle" ] && [ -f "$series_dir/testX.pickle" ] && [ -f "$series_dir/testy.pickle" ]; then
                archive_name="${dataset}_${series_name}_train_and_test.zip"
                echo "  Creating train_and_test archive: $archive_name"
                zip -j "train_test/$archive_name" "$series_dir/trainX.pickle" "$series_dir/trainy.pickle" "$series_dir/testX.pickle" "$series_dir/testy.pickle"
            else
                echo "  Train/test files missing for $series_name. Skipping train_and_test archive."
            fi
        fi
    done
done

echo "Compression complete. Archives are available in the subdirectories: shap, lime, anchor, train_test."

