DATASET_ROOT="dataset/ImageNet1K/val"

python run_DiVA_Patch_common.py --K 100 --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type non_targeted
python run_DiVA_Patch_common.py --K 100 --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 43 --attack_type non_targeted
python run_DiVA_Patch_common.py --K 100 --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 44 --attack_type non_targeted
python run_DiVA_Patch_common.py --K 100 --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type non_targeted

python run_PatchRS.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type non_targeted
python run_PatchRS.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 43 --attack_type non_targeted
python run_PatchRS.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 44 --attack_type non_targeted
python run_PatchRS.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type non_targeted

python run_CamoPatch.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type non_targeted
python run_CamoPatch.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 43 --attack_type non_targeted
python run_CamoPatch.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 44 --attack_type non_targeted
python run_CamoPatch.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type non_targeted
