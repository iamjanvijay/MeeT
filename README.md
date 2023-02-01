# MeeT

This repo contains codes for the following paper:

Janvijay Singh, Fan Bai, Zhen Wang: [Entity Tracking via Effective Use of Multi-Task Learning Model and Mention-guided Decoding](https://arxiv.org/pdf/2210.06444.pdf), EACL 2023

# ProPara Dataset

# Train + Inference using MeeT

```
# clean the existing logs and saved predictions
rm -rf logs/propara/train/state/*
rm -rf logs/propara/train/location/*

# setup a virtual environment - make sure to have exact versions of packages to ensure reproducibility
python3 -m venv ../meet_venv
source ../meet_venv/bin/activate
pip install -r requirements.txt

# train MeeT for state and location question
python code/propara/train/state.py --batch_size 16 --learning_rate 1e-4 --epochs 5 --tokeniser t5-large --lm_model t5-large --output_dir ./logs/propara/train/state;
python code/propara/train/location.py --batch_size 16 --learning_rate 1e-4 --epochs 5 --tokeniser t5-large --lm_model t5-large --output_dir ./logs/propara/train/location;

# CRF based post-processing and evaluation
./eval/propara/merged_state_location_eval.sh logs/propara/train/state logs/propara/train/location;
```

# Inference using MeeT
We release our pre-trained checkpoints in this [Google Drive](https://drive.google.com/drive/folders/1a9SIgd2lYKjLuvisFpENUW1ZvcGeLAiq?usp=sharing). For inference only mode, download the state and location checkpoints and place them in respectively at `logs/propara/inference/state/ckpts` and `logs/propara/inference/location/ckpts` for state and location checkpoints respectively. 

Ensure that the path to placed checkpoints should be `logs/propara/inference/state/ckpts/best.ckpt` and `logs/propara/inference/location/ckpts/best.ckpt`.

```
# clean the existing logs and saved predictions
rm -rf logs/propara/inference/state/*
rm -rf logs/propara/inference/location/*

# setup a virtual environment - make sure to have exact versions of packages to ensure reproducibility
python3 -m venv ../meet_venv;
source ../meet_venv/bin/activate;
pip install -r requirements.txt;

# inference using MeeT for state and location questions using pre-trained checkpoints
python code/propara/inference/state.py --batch_size 16 --tokeniser t5-large --lm_model logs/propara/inference/state/ckpts/best.ckpt --output_dir ./logs/propara/inference/state;
python code/propara/inference/location.py --batch_size 16 --tokeniser t5-large --lm_model logs/propara/inference/location/ckpts/best.ckpt --output_dir ./logs/propara/inference/location;

# CRF based post-processing and evaluation
./eval/propara/merged_state_location_eval.sh logs/propara/inference/state logs/propara/inference/location;
```

# Reproduced Results
```
Best dev results:
=================================================
Question     Avg. Precision  Avg. Recall  Avg. F1
-------------------------------------------------
Inputs                0.859        0.837    0.848
Outputs               0.771        0.957    0.854
Conversions           0.697        0.588    0.638
Moves                 0.764        0.462    0.576
-------------------------------------------------
Overall Precision 0.773                          
Overall Recall    0.711                          
Overall F1        0.741                          
=================================================

Evaluated 43 predictions against 43 answers.

Saved final dev predictions at: eval/propara/final_preds/dev_merged_state_and_location.tsv
Best test results:
=================================================
Question     Avg. Precision  Avg. Recall  Avg. F1
-------------------------------------------------
Inputs                0.886        0.799    0.840
Outputs               0.829        0.892    0.859
Conversions           0.728        0.568    0.638
Moves                 0.769        0.424    0.547
-------------------------------------------------
Overall Precision 0.803                          
Overall Recall    0.671                          
Overall F1        0.731                          
=================================================

Evaluated 54 predictions against 54 answers.

Saved final test predictions at: eval/propara/final_preds/dev_merged_state_and_location.tsv
```

# Recipes Dataset