import pickle as p
import argparse
import os 
import json
import torch
import numpy as np
from utils import (
    NumpyEncoder,
    PerceptualMetricTracker,
    first_success_query_from_process,
    set_seed,
)
from PIL import Image
import matplotlib.pyplot as plt

from attack_methods.PatchRS import PatchRS
from utils.LossFunctions import UnTargeted, Targeted
from models.ImageNetModels_realistic import ImageNetModel


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

    P_INIT = 0.4
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
    metric_tracker = PerceptualMetricTracker(device=device, data_range=255.0)

    save_folder = f"exp_result/PatchRS-{vision_model}-SEED_{SEED}-realistic-{ATTACK_TYPE}"
    os.makedirs(save_folder, exist_ok=True)

    os.makedirs(save_folder + '/processes', exist_ok=True)
    os.makedirs(save_folder + '/results', exist_ok=True)
    os.makedirs(save_folder + '/examples', exist_ok=True)

    for i, path_img in enumerate(path_labels):
        save_file = path_img.replace('.JPEG', '').replace("/", "_")
        process_path = f"{save_folder}/processes/{save_file}.p"
        result_json = f"{save_folder}/results/{save_file}.json"
        if os.path.exists(result_json):
            with open(result_json) as file:
                existing_result = json.load(file)
            if (
                metric_tracker.is_complete(existing_result)
                and "first_success_query" in existing_result
            ):
                metric_tracker.add_summary(existing_result)
                metric_tracker.print_result(path_img, existing_result)
                adversarial.append(existing_result["adversarial"])
                L2.append(existing_result["l2_distance"])
                continue
        print(f'Image #{i + 1}: {path_img}')
        image_dir = os.path.join(dataset_root, path_img)

        true_label = path_labels[path_img]['true_label']
        target_label = path_labels[path_img]['target_label']

        if ATTACK_TYPE == 'non_targeted':
            loss = UnTargeted(model, true_label, to_pytorch=True, device=device)
        else:
            loss = Targeted(model, true_label, target_label, to_pytorch=True, device=device)

        img_cls = Image.open(image_dir).convert("RGB")
        img_cls = np.array(img_cls, dtype='float32')

        if img_cls.shape[-1] != 3:
            continue

        if os.path.exists(process_path):
            with open(process_path, "rb") as file:
                process = p.load(file)
        else:
            init_pred_label = loss.get_label(img_cls)
            if init_pred_label != true_label:
                continue

            set_seed(SEED)
            attacker = PatchRS(img_cls=img_cls, loss_function=loss, max_query=MAX_QUERY, p_init=P_INIT, patch_size=[S, S], update_loc_period=LI)
            attacker.run()
            process = attacker.process
            with open(process_path, "wb") as file:
                p.dump(process, file)

        img_adv = img_cls.copy()
        loc_x, loc_y = process[-1][2]
        patch = process[-1][3]
        img_adv[loc_x:loc_x + S, loc_y:loc_y + S, :] = patch
        metrics = metric_tracker.compute(img_cls, img_adv)

        L2.append(process[-1][-2])
        adversarial.append(process[-1][1])

        summary = {
            "adversarial": process[-1][1],
            "l2_distance": process[-1][-2],
            "location": process[-1][2],
            "loss": process[-1][-1],
            "first_success_query": first_success_query_from_process(
                process, success_index=1, query_index=0
            ),
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
        }
        metric_tracker.print_result(path_img, summary)
        with open(result_json, "w") as file:
            json.dump(summary, file, indent=4, cls=NumpyEncoder)

        im = Image.fromarray(img_adv.astype(np.uint8))
        im.save(f"{save_folder}/examples/{adversarial[-1]}_{path_img.replace('/', '_')}")

        # break
    asr = np.round(sum(adversarial) / len(adversarial), 4) * 100
    mean_l2, std_l2 = np.mean(L2), np.std(L2)
    print(f'Average Attack Success Rate: {asr:.2f}')
    print(f'L2 (mean, std): {mean_l2:.2f} ({std_l2:.2f})')
    metric_tracker.print_summary()
