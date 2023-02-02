#!/bin/bash

# Sample Usage: takes input as ground truth labels and predictions
# ./loc_change_eval.sh ../../fan/data/test_recipes_task_koala_split.json ../../fan/t5_small_recipes_state/test_predictions_32.tsv ../../fan/t5_small_recipes_location/test_predictions_13.tsv

echo "Reading ground-truth from: $1"
echo "Reading state predictions from: $2"
echo "Reading location predictions from: $3"
python eval/recipes/evaluate.py --loc_preds "$3" --state_preds "$2" --gold_json "$1"
