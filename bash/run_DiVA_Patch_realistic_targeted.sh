DATASET_ROOT="dataset/ImageNet1K/val"

python run_DiVA_Patch_realistic.py --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type targeted
python run_DiVA_Patch_realistic.py --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 43 --attack_type targeted
python run_DiVA_Patch_realistic.py --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 44 --attack_type targeted
python run_DiVA_Patch_realistic.py --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type targeted

# python run_DiVA_Patch_realistic.py --vision_model ResNet50 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type targeted
# python run_DiVA_Patch_realistic.py --vision_model ResNet50 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 43 --attack_type targeted
# python run_DiVA_Patch_realistic.py --vision_model ResNet50 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 44 --attack_type targeted
# python run_DiVA_Patch_realistic.py --vision_model ResNet50 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type targeted

# python run_DiVA_Patch_realistic.py --vision_model ViT16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type targeted
# python run_DiVA_Patch_realistic.py --vision_model ViT16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 43 --attack_type targeted
# python run_DiVA_Patch_realistic.py --vision_model ViT16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 44 --attack_type targeted
# python run_DiVA_Patch_realistic.py --vision_model ViT16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type targeted