DATASET_ROOT="dataset/ImageNet1K/val"

python run_DiVA_Patch.py --setting realistic --K 1 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 42 --attack_type non_targeted
python run_DiVA_Patch.py --setting realistic --K 1 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 43 --attack_type non_targeted
python run_DiVA_Patch.py --setting realistic --K 1 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 45 --attack_type non_targeted
python run_DiVA_Patch.py --setting realistic --K 1 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 47 --attack_type non_targeted

python run_DiVA_Patch.py --setting realistic --K 10 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 42 --attack_type non_targeted
python run_DiVA_Patch.py --setting realistic --K 10 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 43 --attack_type non_targeted
python run_DiVA_Patch.py --setting realistic --K 10 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 45 --attack_type non_targeted
python run_DiVA_Patch.py --setting realistic --K 10 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 47 --attack_type non_targeted

python run_DiVA_Patch.py --setting realistic --K 50 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 42 --attack_type non_targeted
python run_DiVA_Patch.py --setting realistic --K 50 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 43 --attack_type non_targeted
python run_DiVA_Patch.py --setting realistic --K 50 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 45 --attack_type non_targeted
python run_DiVA_Patch.py --setting realistic --K 50 --vision_model VGGNet16 --dataset_root $DATASET_ROOT$ --device 'cuda' --seed 47 --attack_type non_targeted
