#!/bin/bash

# Sample Usage: takes input as ground truth labels and predicted states and predicted locations
# ./script_name.sh propara_eval_files/answers.tsv state_predictions.tsv location_predictions.tsv

echo "Ensure that logs folder don't end with /"
echo "state-prediction logs folder: $1"
echo "location-prediction logs folder: $2"

python eval/propara/post_processing_scripts/combine_state_preds_with_log_probs.py $1;
python eval/propara/post_processing_scripts/search_implicit_explicit_and_save_best.py $1 $2;
