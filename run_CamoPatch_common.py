import os 
import json
import torch
import argparse
import numpy as np

from torchvision import transforms

from PIL import Image
from utils import set_seed, pytorch_switch

from utils.LossFunctions import UnTargeted, Targeted

from models.ImageNetModels import ImageNetModel
from attack_methods.CamoPatch_common import CamoPatch_common as CamoPatch
from attack_methods.CamoPatch_common import render

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=100, help='number of semi-transparent circles')
    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--vision_model", type=str, default='VGGNet16', choices=['VGGNet16', 'ResNet50', 'ViT16'])
    parser.add_argument("--device", type=str, default='cuda', help='cuda/cpu')
    parser.add_argument("--dataset_root", type=str, help='ImageNet1K path')
    parser.add_argument("--attack_type", type=str, default='non_targeted', help='targeted/non_targeted')

    parser.add_argument('--demo', help='attack on some example images', action='store_true')

    args = parser.parse_args()

    device = args.device
    vision_model = args.vision_model

    if vision_model == 'VGGNet16':
        model = ImageNetModel(0, device)
    elif vision_model == 'ResNet50':
        model = ImageNetModel(1, device)
    elif vision_model == 'ViT16':
        model = ImageNetModel(2, device)
    else:
        raise NotImplemented

    load_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    N = args.N
    TEMP = 300
    MUT = 0.3
    S = 40
    MAX_QUERY = args.max_query
    LI = 4

    SEED = args.seed
    set_seed(SEED)

    ATTACK_TYPE = args.attack_type

    dataset_root = args.dataset_root
    if args.demo:
        if ATTACK_TYPE == 'targeted':
            path_labels = json.load(open(f'TEST_IMGs/demo_targeted.json'))
        else:
            path_labels = json.load(open(f'TEST_IMGs/demo_untargeted.json'))
    else:
        path_labels = json.load(open(f'TEST_IMGs/{vision_model}_ImgNet1K.json'))

    adversarial, L2 = [], []

    save_folder = f"exp_result/CamoPatch-{vision_model}-SEED_{SEED}-common-{ATTACK_TYPE}"
    os.makedirs(save_folder, exist_ok=True)

    os.makedirs(save_folder + '/processes', exist_ok=True)
    os.makedirs(save_folder + '/examples', exist_ok=True)

    for i, path_img in enumerate(path_labels):
        if os.path.exists(f"{save_folder}/processes/{path_img.replace('/', '_').replace('.JPEG', '.npy')}"):
            continue
        print(f'Image #{i + 1}: {path_img}')
        image_dir = os.path.join(dataset_root, path_img)

        true_label = path_labels[path_img]['true_label']
        target_label = path_labels[path_img]['target_label']
        
        save_file = path_img.replace('.JPEG', '').replace("/", "_")
        save_directory = os.path.join(save_folder + '/processes', save_file)

        if ATTACK_TYPE == 'non_targeted':
            loss = UnTargeted(model, true_label, to_pytorch=True, device=device)
        else:
            loss = Targeted(model, true_label, target_label, to_pytorch=True, device=device)

        img_cls = load_image(Image.open(image_dir).convert("RGB"))
        img_cls = pytorch_switch(img_cls).detach().numpy()

        if img_cls.shape[-1] != 3:
            continue

        init_pred_label = loss.get_label(img_cls)
        if init_pred_label != true_label:
            continue

        params = {
            "x": img_cls,
            "eps": S**2,
            "n_queries": MAX_QUERY,
            "save_directory": save_directory + ".npy",
            "c": img_cls.shape[2],
            "h": img_cls.shape[0],
            "w": img_cls.shape[1],
            "N": N,
            "update_loc_period": LI,
            "mut": MUT,
            "temp": TEMP
        }

        set_seed(SEED)

        attacker = CamoPatch(params, loss, MAX_QUERY)
        attacker.run()
      
        process = attacker.process
        L2.append(process[-1][-2])
        adversarial.append(process[-1][0])

        img_adv = img_cls.copy()
        loc_x, loc_y = process[-1][1]
        patch = render(process[-1][2], S)
        img_adv[loc_x:loc_x + S, loc_y:loc_y + S, :] = patch
        img_adv = img_adv * 255

        im = Image.fromarray(img_adv.astype(np.uint8))
        im.save(f"{save_folder}/examples/{adversarial[-1]}_{path_img.replace('/', '_')}")
        
    asr = np.round(sum(adversarial) / len(adversarial), 4) * 100
    mean_l2, std_l2 = np.mean(L2), np.std(L2)
    print(f'Average Attack Success Rate: {asr:.2f}')
    print(f'L2 (mean, std): {mean_l2:.2f} ({std_l2:.2f})')