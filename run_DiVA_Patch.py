import argparse
import json
import os
import pickle

import numpy as np
from PIL import Image

from utils import (
    NumpyEncoder,
    PerceptualMetricTracker,
    first_success_query_from_process,
    pytorch_switch,
    sample_image_labels,
    set_seed,
)
from utils.LossFunctions import Targeted, UnTargeted


def build_parser(default_setting=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["common", "realistic"], default=default_setting or "realistic", help="common resizes/crops to 224 in [0,1]; realistic uses raw images (pre-processing)")
    parser.add_argument("--exp_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_result"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--vision_model", default="VGGNet16", choices=["VGGNet16", "ResNet50", "ViT16"])
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--dataset_root", required=True, help="ImageNet1K path")
    parser.add_argument("--attack_type", default="non_targeted", choices=["targeted", "non_targeted"])
    parser.add_argument("--N", type=int, default=100, help="number of semi-transparent circles")
    parser.add_argument("--patch_h", type=int, default=40, help="height of patch")
    parser.add_argument("--patch_w", type=int, default=40, help="width of patch")
    parser.add_argument("--grid_w", type=int, default=40, help="width of grid archive")
    parser.add_argument("--grid_h", type=int, default=40, help="height of grid archive")
    parser.add_argument("--delta_max", type=float, default=10.0, help="maximum selection temperature")
    parser.add_argument("--delta_min", type=float, default=0.1, help="minimum selection temperature")
    parser.add_argument("--K", type=int, default=100, help="number of refinement queries")
    parser.add_argument("--save_imgs", action="store_true", help="save adversarial images")
    parser.add_argument("--demo", action="store_true", help="attack example images")
    return parser


def _components(setting):
    if setting == "common":
        from attack_methods.DiVA_Patch_common import DiVA_Patch_common as DiVA_Patch
        from models.ImageNetModels import ImageNetModel
    else:
        from attack_methods.DiVA_Patch import DiVA_Patch
        from models.ImageNetModels_realistic import ImageNetModel
    return DiVA_Patch, ImageNetModel


def _image_loader(setting):
    if setting == "realistic":
        def load_realistic(path):
            return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)

        return load_realistic

    from torchvision import transforms

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    def load_common(path):
        return pytorch_switch(transform(Image.open(path).convert("RGB"))).detach().numpy()

    return load_common


def _save_folder(args):
    experiment = f"DiVA_Patch-GridSize_{args.grid_h}_{args.grid_w}-Delta_{args.delta_max}_{args.delta_min}-K{args.K}-{args.vision_model}-{args.setting}-{args.attack_type}"
    return os.path.join(args.exp_root, experiment, f"SEED_{args.seed}")


def _safe_name(path):
    return path.replace(".JPEG", "").replace("/", "_").replace("\\", "_")


def _build_adversarial(image, location, patch):
    adversarial = image.copy()
    loc_x, loc_y = location
    adversarial[loc_x:loc_x + patch.shape[0], loc_y:loc_y + patch.shape[1], :] = patch
    return adversarial


def main(default_setting=None):
    parser = build_parser(default_setting)
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num_images must be positive")
    if args.max_query <= 0:
        parser.error("--max_query must be positive")
    if args.patch_h <= 0 or args.patch_w <= 0:
        parser.error("--patch_h and --patch_w must be positive")
    if args.grid_h <= 0 or args.grid_w <= 0:
        parser.error("--grid_h and --grid_w must be positive")

    DiVA_Patch, ImageNetModel = _components(args.setting)
    model_index = {"VGGNet16": 0, "ResNet50": 1, "ViT16": 2}[args.vision_model]
    model = ImageNetModel(model_index, args.device)
    load_image = _image_loader(args.setting)
    set_seed(args.seed)

    if args.demo:
        label_file = "TEST_IMGs/demo_targeted.json" if args.attack_type == "targeted" else "TEST_IMGs/demo_untargeted.json"
    else:
        label_file = f"TEST_IMGs/{args.vision_model}_ImgNet1K.json"
    with open(label_file) as file:
        path_labels = json.load(file)

    save_dir = _save_folder(args)
    map_elites_dir = os.path.join(save_dir, "map_elites")
    process_dir = os.path.join(save_dir, "processes")
    result_dir = os.path.join(save_dir, "results")
    example_dir = os.path.join(save_dir, "examples")
    for directory in (map_elites_dir, process_dir, result_dir, example_dir):
        os.makedirs(directory, exist_ok=True)

    path_labels = sample_image_labels(path_labels, args.num_images, args.seed, label_file, os.path.join(save_dir, "sampled_images.json"))
    metric_tracker = PerceptualMetricTracker(device=args.device, data_range=1.0 if args.setting == "common" else 255.0)
    adversarial_results, l2_values, location_counts = [], [], []

    for index, path_img in enumerate(path_labels, start=1):
        save_file = _safe_name(path_img)
        result_json = os.path.join(result_dir, f"{save_file}.json")
        process_path = os.path.join(process_dir, f"{save_file}_process.p")
        if os.path.exists(result_json):
            with open(result_json) as file:
                existing_result = json.load(file)
            if metric_tracker.is_complete(existing_result) and "first_success_query" in existing_result:
                metric_tracker.add_summary(existing_result)
                metric_tracker.print_result(path_img, existing_result)
                adversarial_results.append(existing_result["adversarial"])
                l2_values.append(existing_result["l2_distance"])
                if existing_result["adversarial"]:
                    location_counts.append(len(existing_result.get("#locs", [])))
                continue

        print(f"Image #{index}: {path_img}")
        image_path = os.path.join(args.dataset_root, path_img)
        true_label = path_labels[path_img]["true_label"]
        target_label = path_labels[path_img]["target_label"]
        if args.attack_type == "non_targeted":
            loss = UnTargeted(model, true_label, to_pytorch=True, device=args.device)
        else:
            loss = Targeted(model, true_label, target_label, to_pytorch=True, device=args.device)

        img_cls = load_image(image_path)
        if img_cls.shape[-1] != 3:
            continue

        if os.path.exists(process_path):
            with open(process_path, "rb") as file:
                process = pickle.load(file)
            best = process[-1]
            img_adv = _build_adversarial(img_cls, best[2], np.asarray(best[3]))
            metrics = metric_tracker.compute(img_cls, img_adv)
            summary = {
                "adversarial": best[1],
                "l2_distance": best[4],
                "first_success_query": first_success_query_from_process(process, success_index=1, query_index=0),
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
            }
            metric_tracker.print_result(path_img, summary)
            with open(result_json, "w") as file:
                json.dump(summary, file, indent=4, cls=NumpyEncoder)

            if args.save_imgs:
                output = img_adv * 255.0 if args.setting == "common" else img_adv
                Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(os.path.join(example_dir, f"{best[1]}_{path_img.replace('/', '_')}"))

            adversarial_results.append(best[1])
            l2_values.append(best[4])
            continue

        if loss.get_label(img_cls) != true_label:
            continue

        set_seed(args.seed)
        attacker = DiVA_Patch(img_cls=img_cls, loss_function=loss, delta_max=args.delta_max, delta_min=args.delta_min, K=args.K, patch_size=[args.patch_h, args.patch_w], grid_size=[args.grid_h, args.grid_w], max_query=args.max_query)
        attacker.run()

        map_elites = attacker.return_map_elites()
        with open(os.path.join(map_elites_dir, f"{save_file}_map_elites.p"), "wb") as file:
            pickle.dump(map_elites, file)

        if args.save_imgs:
            process = attacker.process
            with open(process_path, "wb") as file:
                pickle.dump(process, file)

        results = []
        for cell_index, individual in attacker.map_elites.grid_final.items():
            if individual is not None and individual.success_attack:
                results.append([cell_index, individual.success_attack, individual.location, individual.patch, individual.l2, individual.loss])
        with open(os.path.join(result_dir, f"{save_file}_result.p"), "wb") as file:
            pickle.dump(results, file)

        best_idv = attacker.get_best_quality_solution()
        print(f"Best patch:\n+ Adversarial: {best_idv.success_attack}\n+ L2: {best_idv.l2:.2f}")
        if best_idv.success_attack:
            location_counts.append(len(results))

        img_adv = _build_adversarial(img_cls, best_idv.location, best_idv.patch)
        metrics = metric_tracker.compute(img_cls, img_adv)
        summary = {
            "adversarial": best_idv.success_attack,
            "l2_distance": best_idv.l2,
            "#locs": [[result[2], result[4], result[5]] for result in results],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
            "first_success_query": attacker.first_success_query,
        }
        metric_tracker.print_result(path_img, summary)
        with open(result_json, "w") as file:
            json.dump(summary, file, indent=4, cls=NumpyEncoder)

        if args.save_imgs:
            output = img_adv * 255.0 if args.setting == "common" else img_adv
            Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(os.path.join(example_dir, f"{best_idv.success_attack}_{path_img.replace('/', '_')}"))
        l2_values.append(best_idv.l2)
        adversarial_results.append(best_idv.success_attack)

    if not adversarial_results:
        print("No correctly classified, unfinished images were attacked.")
        return
    print(f"Average Attack Success Rate: {100 * np.mean(adversarial_results):.2f}")
    print(f"L2 (mean, std): {np.mean(l2_values):.2f} ({np.std(l2_values):.2f})")
    if location_counts:
        print(f"#Locs (mean, std): {np.mean(location_counts):.2f} ({np.std(location_counts):.2f})")
    metric_tracker.print_summary()


if __name__ == "__main__":
    main()
