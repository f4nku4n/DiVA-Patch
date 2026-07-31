DATASET_PATH="/path/to/ILSVRC2012/val"

python run_DevoPatch.py --setting realistic \
  --vision_model VGGNet16 \
  --dataset_root "$DATASET_PATH" \
  --device cuda \
  --attack_type targeted
