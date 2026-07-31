DATASET_ROOT="demo_imgs"

python run_DiVA_Patch.py --setting realistic --K 100 --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type non_targeted --demo
python run_PatchRS.py --setting realistic --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type non_targeted --demo
python run_CamoPatch.py --setting realistic --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 42 --attack_type non_targeted --demo
