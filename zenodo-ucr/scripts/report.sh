#!/bin/bash
# This script processes a "complete" folder containing the subfolders:
# models, train_test, shap, lime, anchor.
#
# For each series (identified by its model file in models), the script extracts:
#   - dataset (first token: univariate|multivariate)
#   - series name (second token)
#
# It then checks for the presence of corresponding ZIP files in the
# anchor, lime, and shap folders.
#
# Finally, it prints:
#   1. The number of univariate series and the percentages with anchor, lime, shap, and all three.
#   2. The same information for multivariate series.
#   3. A sorted alphabetical list of all series with a concise status for each explainer.
#
# Usage:
#   ./report_complete.sh <complete_folder_path>
# Example:
#   ./report_complete.sh /opt/jupyterhub/fast/shared_dir/UCI-Benchmark/complete

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <complete_folder_path>"
    exit 1
fi

COMPLETE_DIR="$1"
if [ ! -d "$COMPLETE_DIR" ]; then
    echo "Error: Directory $COMPLETE_DIR does not exist."
    exit 1
fi

# Define the subdirectories.
MODELS_DIR="${COMPLETE_DIR}/models"
ANCHOR_DIR="${COMPLETE_DIR}/anchor"
LIME_DIR="${COMPLETE_DIR}/lime"
SHAP_DIR="${COMPLETE_DIR}/shap"

# Initialize counters.
total_univariate=0
total_multivariate=0

univariate_anchor=0
univariate_lime=0
univariate_shap=0
univariate_all=0

multivariate_anchor=0
multivariate_lime=0
multivariate_shap=0
multivariate_all=0

# Array to store detailed records.
records=()

# Process each model file.
for modelPath in "$MODELS_DIR"/*.zip; do
    [ -e "$modelPath" ] || continue
    filename=$(basename "$modelPath")
    base="${filename%.zip}"
    # Assume naming: dataset_series_...
    dataset=$(echo "$base" | cut -d'_' -f1)
    series=$(echo "$base" | cut -d'_' -f2)
    full_series="${dataset}_${series}"
    
    # Expected corresponding file names.
    anchorFile="${ANCHOR_DIR}/${full_series}_anchor_values.zip"
    limeFile="${LIME_DIR}/${full_series}_lime_values.zip"
    shapFile="${SHAP_DIR}/${full_series}_shap_values.zip"

    has_anchor=0
    has_lime=0
    has_shap=0

    [ -f "$anchorFile" ] && has_anchor=1
    [ -f "$limeFile" ]   && has_lime=1
    [ -f "$shapFile" ]   && has_shap=1

    all_three=0
    if [ "$has_anchor" -eq 1 ] && [ "$has_lime" -eq 1 ] && [ "$has_shap" -eq 1 ]; then
        all_three=1
    fi

    # Update counters per dataset.
    if [ "$dataset" == "univariate" ]; then
        total_univariate=$((total_univariate + 1))
        univariate_anchor=$((univariate_anchor + has_anchor))
        univariate_lime=$((univariate_lime + has_lime))
        univariate_shap=$((univariate_shap + has_shap))
        univariate_all=$((univariate_all + all_three))
    elif [ "$dataset" == "multivariate" ]; then
        total_multivariate=$((total_multivariate + 1))
        multivariate_anchor=$((multivariate_anchor + has_anchor))
        multivariate_lime=$((multivariate_lime + has_lime))
        multivariate_shap=$((multivariate_shap + has_shap))
        multivariate_all=$((multivariate_all + all_three))
    fi

    # Build a concise record for detailed list.
    anchor_status=$([[ $has_anchor -eq 1 ]] && echo "Yes" || echo "No")
    lime_status=$([[ $has_lime -eq 1 ]] && echo "Yes" || echo "No")
    shap_status=$([[ $has_shap -eq 1 ]] && echo "Yes" || echo "No")
    record="${full_series}: Anchor:${anchor_status}, Lime:${lime_status}, Shap:${shap_status}"
    records+=("$record")
done

# Function to calculate percentage.
calc_percentage() {
    # $1: count, $2: total
    if [ "$2" -gt 0 ]; then
        echo "scale=2; $1*100/$2" | bc
    else
        echo "0"
    fi
}

echo "Univariate series: $total_univariate"
if [ "$total_univariate" -gt 0 ]; then
    echo "  Anchor: $(calc_percentage $univariate_anchor $total_univariate)%"
    echo "  Lime:   $(calc_percentage $univariate_lime $total_univariate)%"
    echo "  Shap:   $(calc_percentage $univariate_shap $total_univariate)%"
    echo "  All 3:  $(calc_percentage $univariate_all $total_univariate)%"
fi
echo ""
echo "Multivariate series: $total_multivariate"
if [ "$total_multivariate" -gt 0 ]; then
    echo "  Anchor: $(calc_percentage $multivariate_anchor $total_multivariate)%"
    echo "  Lime:   $(calc_percentage $multivariate_lime $total_multivariate)%"
    echo "  Shap:   $(calc_percentage $multivariate_shap $total_multivariate)%"
    echo "  All 3:  $(calc_percentage $multivariate_all $total_multivariate)%"
fi
echo ""
echo "Detailed series list (sorted alphabetically):"
for rec in "${records[@]}"; do
    echo "$rec"
done | sort
