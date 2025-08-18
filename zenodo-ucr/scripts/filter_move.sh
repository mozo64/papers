#!/bin/bash
# This script creates a "complete" folder structure with subdirectories:
# complete/models, complete/train_test, complete/shap, complete/lime, complete/anchor.
#
# For each series (identified by a model file in models), it does the following:
#   - Extracts the dataset and series name from the filename (first two tokens).
#   - Checks for the corresponding train_test file (required) and optional files:
#         shap:   {dataset}_{series}_shap_values.zip
#         lime:   {dataset}_{series}_lime_values.zip
#         anchor: {dataset}_{series}_anchor_values.zip
#   - Awards 1 point for each optional file found.
#   - Moves all the files (model + corresponding files) into the "complete" folder
#     if train_test exists AND the total optional points are at least 2.
#
# Log output is printed in a concise format.
#
# Usage: run this script in the directory that contains the folders:
#       models, train_test, shap, lime, anchor

# Create the complete folder structure.
mkdir -p complete/{models,train_test,shap,lime,anchor}

# Loop over each model file.
for modelPath in models/*.zip; do
    base=$(basename "$modelPath")            # e.g. univariate_Worms_model_tf.zip
    baseNoExt="${base%.zip}"                  # e.g. univariate_Worms_model_tf
    # Split on underscore; the first token is dataset, the second is series.
    IFS='_' read -r dataset series rest <<< "$baseNoExt"
    
    # Build expected corresponding file names.
    ttFile="${dataset}_${series}_train_and_test.zip"
    shapFile="${dataset}_${series}_shap_values.zip"
    limeFile="${dataset}_${series}_lime_values.zip"
    anchorFile="${dataset}_${series}_anchor_values.zip"
    
    # Check existence.
    if [ -f "train_test/$ttFile" ]; then
         tStatus="OK"
    else
         tStatus="FAIL"
    fi
    
    # Initialize optional points.
    points=0
    if [ -f "shap/$shapFile" ]; then
         sStatus="OK"
         points=$(( points + 1 ))
    else
         sStatus="FAIL"
    fi
    
    if [ -f "lime/$limeFile" ]; then
         lStatus="OK"
         points=$(( points + 1 ))
    else
         lStatus="FAIL"
    fi
    
    if [ -f "anchor/$anchorFile" ]; then
         aStatus="OK"
         points=$(( points + 1 ))
    else
         aStatus="FAIL"
    fi

    # Concise log message.
    logLine="${dataset}_${series}: TT:$tStatus, SH:$sStatus, LI:$lStatus, AN:$aStatus (Pts:$points)"
    
    # Decision: move if train_test exists and at least 2 optional files are present.
    if [ "$tStatus" == "OK" ] && [ "$points" -ge 2 ]; then
         echo "$logLine -> Moved"
         mv "$modelPath" complete/models/
         mv "train_test/$ttFile" complete/train_test/
         [ -f "shap/$shapFile" ]   && mv "shap/$shapFile"   complete/shap/
         [ -f "lime/$limeFile" ]   && mv "lime/$limeFile"   complete/lime/
         [ -f "anchor/$anchorFile" ] && mv "anchor/$anchorFile" complete/anchor/
    else
         echo "$logLine -> Not moved"
    fi

done
