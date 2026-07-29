DATASET_ROOT="demo_imgs"

python run_DiVA_Patch_realistic.py --K 100 --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type targeted --demo
python run_PatchRS_realistic.py --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type targeted --demo
python run_CamoPatch_realistic.py --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type targeted --demo
