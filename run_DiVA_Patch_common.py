import os
import json
import argparse
import numpy as np
import pickle as p
from torchvision import transforms

from PIL import Image
from utils import (
    NumpyEncoder,
    PerceptualMetricTracker,
    pytorch_switch,
    set_seed,
)

from utils.LossFunctions import UnTargeted, Targeted

from models.ImageNetModels import ImageNetModel
from attack_methods.DiVA_Patch_common import DiVA_Patch_common as DiVA_Patch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--vision_model", type=str, default='VGGNet16', choices=['VGGNet16', 'ResNet50', 'ViT16'])
    parser.add_argument("--device", type=str, default='cuda', help='cuda/cpu')
    parser.add_argument("--dataset_root", type=str, help='ImageNet1K path')
    parser.add_argument("--attack_type", type=str, default='non_targeted', help='targeted/non_targeted')

    parser.add_argument("--N", type=int, default=100, help='number of semi-transparent circles')
    parser.add_argument("--patch_h", type=int, default=40, help='height of patch')
    parser.add_argument("--patch_w", type=int, default=40, help='width of patch')
    parser.add_argument("--grid_w", type=int, default=40, help='width of grid (archive)')
    parser.add_argument("--grid_h", type=int, default=40, help='width of grid (archive)')

    parser.add_argument("--delta_max", type=float, default=10.0, help='delta max')
    parser.add_argument("--delta_min", type=float, default=0.1, help='delta min')
    parser.add_argument("--K", type=int, default=100, help='number of refinement queries')

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
    PATCH_SIZE = [args.patch_h, args.patch_w]
    GRID_SIZE = [args.grid_h, args.grid_w]
    MAX_QUERY = args.max_query

    K = args.K
    DELTA_MAX, DELTA_MIN = args.delta_max, args.delta_min

    ATTACK_TYPE = args.attack_type

    SEED = args.seed
    set_seed(SEED)

    dataset_root = args.dataset_root
    
    if args.demo:
        if ATTACK_TYPE == 'targeted':
            path_labels = json.load(open(f'TEST_IMGs/demo_targeted.json'))
        else:
            path_labels = json.load(open(f'TEST_IMGs/demo_untargeted.json'))
    else:
        path_labels = json.load(open(f'TEST_IMGs/{vision_model}_ImgNet1K.json'))

    adversarial, L2, locs = [], [], []
    metric_tracker = PerceptualMetricTracker(device=device, data_range=1.0)

    save_dir = f"exp_result/DiVA_Patch-GridSize_{args.grid_h}_{args.grid_w}-Delta_{DELTA_MAX}_{DELTA_MIN}-K{args.K}-{vision_model}-SEED_{SEED}-common-{ATTACK_TYPE}"
    os.makedirs(save_dir, exist_ok=True)

    os.makedirs(save_dir + '/map_elites', exist_ok=True)
    os.makedirs(save_dir + '/processes', exist_ok=True)
    os.makedirs(save_dir + '/results', exist_ok=True)
    os.makedirs(save_dir + '/examples', exist_ok=True)

    for i, path_img in enumerate(path_labels):
        result_json = f"{save_dir}/results/{path_img.replace('/', '_').replace('.JPEG', '')}.json"
        process_path = f"{save_dir}/processes/{path_img.replace('/', '_').replace('.JPEG', '')}_process.p"
        if os.path.exists(result_json):
            with open(result_json) as file:
                existing_result = json.load(file)
            if metric_tracker.is_complete(existing_result):
                metric_tracker.add_summary(existing_result)
                adversarial.append(existing_result["adversarial"])
                L2.append(existing_result["l2_distance"])
                if existing_result["adversarial"]:
                    locs.append(len(existing_result.get("#locs", [])))
                continue
        attack_result = {}
        print(f'Image #{i + 1}: {path_img}')
        image_dir = os.path.join(dataset_root, path_img)

        save_file = path_img.replace('.JPEG', '').replace("/", "_")
        save_directory = os.path.join(save_dir, save_file)

        true_label = path_labels[path_img]['true_label']
        target_label = path_labels[path_img]['target_label']

        if ATTACK_TYPE == 'non_targeted':
            loss = UnTargeted(model, true_label, to_pytorch=True, device=device)
        else:
            loss = Targeted(model, true_label, target_label, to_pytorch=True, device=device)

        img_cls = load_image(Image.open(image_dir).convert("RGB"))
        img_cls = pytorch_switch(img_cls).detach().numpy()

        if img_cls.shape[-1] != 3:
            continue

        if os.path.exists(process_path):
            with open(process_path, "rb") as file:
                process = p.load(file)
            best = process[-1]
            img_adv = img_cls.copy()
            loc_x, loc_y = best[2]
            patch = best[3]
            img_adv[
                loc_x:loc_x + patch.shape[0],
                loc_y:loc_y + patch.shape[1],
                :,
            ] = patch
            metrics = metric_tracker.compute(img_cls, img_adv)
            attack_result = existing_result if os.path.exists(result_json) else {}
            attack_result.update({
                "adversarial": best[1],
                "l2_distance": best[4],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
            })
            with open(result_json, "w") as file:
                json.dump(attack_result, file, indent=4, cls=NumpyEncoder)
            adversarial.append(best[1])
            L2.append(best[4])
            continue

        init_pred_label = loss.get_label(img_cls)
        if init_pred_label != true_label:
            continue

        set_seed(SEED)

        attacker = DiVA_Patch(img_cls=img_cls, loss_function=loss,
                              delta_max=DELTA_MAX, delta_min=DELTA_MIN, K=K,
                              patch_size=PATCH_SIZE, grid_size=GRID_SIZE,
                              max_query=MAX_QUERY)
        attacker.run()

        map_elites = attacker.return_map_elites()
        p.dump(map_elites,
               open(f"{save_dir}/map_elites/{path_img.replace('/', '_').replace('.JPEG', '')}_map_elites.p", 'wb'))

        process = attacker.process
        p.dump(process, open(f"{save_dir}/processes/{path_img.replace('/', '_').replace('.JPEG', '')}_process.p", 'wb'))

        results = []
        for idx, idv in attacker.map_elites.grid_final.items():
            if idv is not None and idv.success_attack:
                results.append([idx, idv.success_attack, idv.location, idv.patch, idv.l2, idv.loss])

        p.dump(results, open(f"{save_dir}/results/{path_img.replace('/', '_').replace('.JPEG', '')}_result.p", 'wb'))

        best_idv = attacker.get_best_quality_solution()
        print('Best patch:\n', f'+ Adversarial: {best_idv.success_attack}\n', f'+ L2: {best_idv.l2:.2f}')

        if best_idv.success_attack:
            locs.append(len(results))

        img_adv = img_cls.copy()
        loc_x, loc_y = best_idv.location
        s = best_idv.s
        patch = best_idv.patch
        img_adv[loc_x:loc_x + s[0], loc_y:loc_y + s[1], :] = patch
        metrics = metric_tracker.compute(img_cls, img_adv)

        img_adv = img_adv * 255
        
        im = Image.fromarray(img_adv.astype(np.uint8))
        im.save(f"{save_dir}/examples/{best_idv.success_attack}_{path_img.replace('/', '_')}")

        attack_result['adversarial'] = best_idv.success_attack
        attack_result['l2_distance'] = best_idv.l2
        attack_result['#locs'] = [[x[2], x[4], x[5]] for x in results]
        attack_result['ssim'] = metrics['ssim']
        attack_result['lpips'] = metrics['lpips']

        json.dump(attack_result,
                  open(f"{save_dir}/results/{path_img.replace('/', '_').replace('.JPEG', '')}.json", 'w'), indent=4,
                  cls=NumpyEncoder)

        L2.append(best_idv.l2)
        adversarial.append(best_idv.success_attack)

    asr = np.round(sum(adversarial) / len(adversarial), 4) * 100
    mean_l2, std_l2 = np.mean(L2), np.std(L2)
    print(f'Average Attack Success Rate: {asr:.2f}')
    print(f'L2 (mean, std): {mean_l2:.2f} ({std_l2:.2f})')
    print(f'#Locs (mean, std): {np.mean(locs):.2f} ({np.std(locs):.2f})')
    metric_tracker.print_summary()

