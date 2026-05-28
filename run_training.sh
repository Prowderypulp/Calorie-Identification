#!/bin/bash
# Script to run the full fine-tuning pipeline on the downloaded datasets

echo "Starting Full Fine-Tuning Pipeline..."

# Activate the conda environment
source /home/drtex/anaconda3/bin/activate mlenv

echo "============================================="
echo "1. Training Classifier on Food-101"
echo "============================================="
python scripts/train_classifier.py --data_dir ./data/food-101 --use_torchvision_dataset --output_dir ./app/models --epochs 20 --batch_size 32

echo "============================================="
echo "2. Training Portion Estimator on Nutrition5k"
echo "============================================="
# Note: Nutrition5k metadata processing would need to match the specific CSVs downloaded.
# The metadata files are: dish_metadata_cafe1.csv, dish_metadata_cafe2.csv
# Assuming a script merges them into train.csv and val.csv in the future.
# For now, this points to the dummy paths or requires manual CSV creation based on the actual Nutrition5k split.
# I will output a message to ensure the CSVs are prepared first.
echo "Make sure to prepare train_csv and val_csv from Nutrition5k metadata before running this step."
# python scripts/train_portion.py --train_csv ./data/nutrition5k/train.csv --val_csv ./data/nutrition5k/val.csv --img_dir ./data/nutrition5k/imagery --output_dir ./app/models --epochs 20 --batch_size 16

echo "============================================="
echo "3. Exporting to ONNX"
echo "============================================="
python scripts/export_onnx.py --model_path ./app/models/classifier.pth --model_type classifier --num_classes 101 --output_path ./app/models/classifier.onnx
python scripts/export_onnx.py --model_path ./app/models/portion_estimator.pth --model_type portion --output_path ./app/models/portion_estimator.onnx

echo "Pipeline Complete!"
