DATASET_ROOT="demo_imgs"

python run_DiVA_Patch.py --setting realistic --K 100 --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type targeted --demo
python run_PatchRS.py --setting realistic --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type targeted --demo
python run_CamoPatch.py --setting realistic --vision_model VGGNet16 --dataset_root "$DATASET_ROOT" --device 'cuda' --seed 45 --attack_type targeted --demo
