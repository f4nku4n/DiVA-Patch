DATASET_ROOT="demo_imgs"

python run_DiVA_Patch_realistic.py --K 100 --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type non_targeted --demo
python run_PatchRS_realistic.py --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type non_targeted --demo
python run_CamoPatch_realistic.py --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type non_targeted --demo
