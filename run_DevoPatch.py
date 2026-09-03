import os
import json
import argparse
import pickle as p

import numpy as np
from PIL import Image

from utils import NumpyEncoder, PerceptualMetricTracker
from utils import select_devopatch_target,sample_image_labels, set_seed

from attack_methods.DevoPatch import DevoPatch

from utils.LossFunctions import Targeted, UnTargeted
from factory import getVisionModel, getImageLoader, getLabelFile

VALID_MODELS = {
    "ImageNet1K": ("VGGNet16", "ResNet50", "ViT16"),
    "Flower102": ("EfficientNetV2S",),
    "Food101": ("Swin",),
}

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["common", "realistic"],
                        default="realistic",
                        help="common resizes/crops to 224 in [0,1]; realistic uses raw images (pre-processing)")
    parser.add_argument("--exp_root", default="./exp_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--dataset", choices=list(VALID_MODELS), default="ImageNet1K")
    parser.add_argument("--vision_model", default="VGGNet16", choices=["VGGNet16", "ResNet50", "ViT16", "EfficientNetV2S", "Swin"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--attack_type", default="non_targeted", choices=["targeted", "non_targeted"])
    parser.add_argument("--pop_size", type=int, default=10)
    parser.add_argument("--init_rate", type=float, default=0.35)
    parser.add_argument("--mutation_rate", type=int, default=1)
    parser.add_argument("--fitness_norm", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--save_imgs", action="store_true", help="save adversarial images")
    parser.add_argument("--save_process", action="store_true", help="save attack process")
    parser.add_argument("--demo", action="store_true")
    return parser



def _safe_name(path):
    return path.replace("\\", "_").replace("/", "_").rsplit(".", 1)[0]


def _load_target(path, setting, source_shape, load_image):
    if setting == "common":
        return load_image(path)
    image = Image.open(path).convert("RGB")
    height, width = source_shape[:2]
    image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def _load_target_manifest(path, seed, label_file):
    expected = {"seed": int(seed), "label_source": os.path.normpath(label_file)}
    if os.path.exists(path):
        with open(path) as file:
            manifest = json.load(file)
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"target manifest conflict for {key}: expected {value!r}, found {manifest.get(key)!r}")
        manifest.setdefault("targets", {})
        return manifest
    return {**expected, "targets": {}}


def _save_target_manifest(path, manifest):
    with open(path, "w") as file:
        json.dump(manifest, file, indent=4)


def _rebuild_from_process(source, process):
    best = process[-1]
    top, left = best[2]
    patch = np.asarray(best[3])
    adversarial = source.copy()
    adversarial[top:top + patch.shape[0], left:left + patch.shape[1]] = patch
    return adversarial


def _print_result(image, summary):
    first_success = summary.get("first_success_query")
    first_success = "N/A" if first_success is None else first_success
    print(
        f"{image} | Success: {bool(summary['adversarial'])} | "
        f"Queries: {int(summary['queries'])} | "
        f"First success query: {first_success} | "
        f"Patch area: {int(summary['patch_area'])} "
        f"({100.0 * float(summary['patch_area_ratio']):.4f}%) | "
        f"L2: {float(summary['l2_distance']):.6f} | "
        f"SSIM: {float(summary['ssim']):.6f} | "
        f"LPIPS-AlexNet: {float(summary['lpips']):.6f}"
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load vision model
    VisionModel, model_args = getVisionModel(args.dataset, args.vision_model)
    model = VisionModel(*model_args, args.device)

    # Load label file
    label_file = getLabelFile(args)
    all_labels = json.load(open(label_file))

    # Create results folders
    experiment = (f"DevoPatch-Pop_{args.pop_size}-Init_{args.init_rate}-Mutation_{args.mutation_rate}-"
                  f"Norm_{args.fitness_norm}-{args.vision_model}-{args.setting}-{args.attack_type}")
    save_dir = f'{args.exp_root}/{experiment}/SEED_{args.seed}'

    result_dir = f'{save_dir}/results'
    example_dir = f'{save_dir}/examples'
    process_dir = f'{save_dir}/processes'
    for directory in (save_dir, process_dir, result_dir, example_dir):
        os.makedirs(directory, exist_ok=True)

    path_labels = sample_image_labels(all_labels, args.num_images, args.seed, label_file, f'{save_dir}/sampled_images.json')

    target_manifest_path = f"{save_dir}/target_images.json"
    target_manifest = _load_target_manifest(target_manifest_path, args.seed, label_file)

    metric_tracker = PerceptualMetricTracker(device=args.device, data_range=1.0 if args.setting == "common" else 255.0)

    load_image = getImageLoader(args.dataset, args.setting)
    set_seed(args.seed)

    successes, l2_values, areas, query_values = [], [], [], []
    for index, path_img in enumerate(path_labels, start=1):
        print(f'Image #{index}: {path_img}')
        save_file = _safe_name(path_img)
        result_path = f'{result_dir}/{save_file}.json'
        process_path = f'{process_dir}/{save_file}_process.p'
        final_result_path = f'{result_dir}/{save_file}_result.p'

        if os.path.exists(result_path):
            summary = json.load(open(result_path))
            if (
                metric_tracker.is_complete(summary)
                and "first_success_query" in summary
                and "target_image" in summary
                and os.path.exists(final_result_path)
            ):
                metric_tracker.add_summary(summary)
                _print_result(path_img, summary)
                successes.append(bool(summary["adversarial"]))
                l2_values.append(float(summary["l2_distance"]))
                areas.append(float(summary["patch_area_ratio"]))
                query_values.append(int(summary["queries"]))
                continue

        target_entry = target_manifest["targets"].get(path_img)
        if target_entry is None:
            target_path, target_class = select_devopatch_target(
                all_labels, path_img, args.attack_type == "targeted", args.seed,
            )
            target_entry = {"target_image": target_path, "target_class": target_class}
            target_manifest["targets"][path_img] = target_entry
            _save_target_manifest(target_manifest_path, target_manifest)
        target_path = target_entry["target_image"]
        target_class = int(target_entry["target_class"])

        source_file = f'{args.dataset_root}/{path_img}'
        target_file = f'{args.dataset_root}/{target_path}'

        source = load_image(source_file)
        target = _load_target(target_file, args.setting, source.shape, load_image)
        true_label = int(path_labels[path_img]['true_label'])
        configured_target = int(path_labels[path_img]['target_label'])
        if args.attack_type == "targeted":
            loss = Targeted(model, true_label, configured_target, to_pytorch=True, device=args.device)
        else:
            loss = UnTargeted(model, true_label, to_pytorch=True, device=args.device)

        if loss.get_label(source) != true_label:
            print(f"Skip {path_img}: clean image is misclassified")
            continue

        if os.path.exists(process_path):
            process = p.load(open(process_path, 'rb'))
            adversarial = _rebuild_from_process(source, process)
            best = process[-1]
            patch = np.asarray(best[3])
            final_patch = patch
            summary = {
                "adversarial": bool(best[1]),
                "l2_distance": float(best[4]),
                "fitness": float(best[5]),
                "queries": int(best[0]),
                "first_success_query": next((int(record[0]) for record in process if record[1]), None),
                "location": best[2],
                "rectangle": [
                    best[2][0], best[2][1],
                    best[2][0] + patch.shape[0], best[2][1] + patch.shape[1],
                ],
                "patch_area": int(patch.shape[0] * patch.shape[1]),
                "patch_area_ratio": float(patch.shape[0] * patch.shape[1] / (source.shape[0] * source.shape[1])),
                **target_entry,
            }
        else:
            set_seed(args.seed)
            attacker = DevoPatch(
                source, target, loss,
                pop_size=args.pop_size, init_rate=args.init_rate, mutation_rate=args.mutation_rate,
                fitness_norm=args.fitness_norm, max_query=args.max_query
            )
            attacker.run()
            result = attacker.get_best()
            final_patch = np.asarray(result["patch"])
            adversarial = attacker.build_adversarial()
            if args.save_process:
                process = attacker.process
                p.dump(process, open(process_path, 'wb'))
            summary = {
                "adversarial": result["success"],
                "l2_distance": result["l2"],
                "fitness": result["fitness"],
                "queries": result["queries"],
                "first_success_query": result["first_success_query"],
                "location": result["location"],
                "rectangle": result["rectangle"],
                "patch_area": result["patch_area"],
                "patch_area_ratio": result["patch_area_ratio"],
                **target_entry,
            }

        final_result = {
            "adversarial": bool(summary["adversarial"]),
            "location": summary["location"],
            "rectangle": summary["rectangle"],
            "patch": final_patch.copy(),
            "l2_distance": float(summary["l2_distance"]),
            "fitness": float(summary["fitness"]),
            "queries": int(summary["queries"]),
            "first_success_query": summary["first_success_query"],
            "patch_area": int(summary["patch_area"]),
            "patch_area_ratio": float(summary["patch_area_ratio"]),
            "setting": args.setting,
            "attack_type": args.attack_type,
            "pop_size": args.pop_size,
            "init_rate": args.init_rate,
            "mutation_rate": args.mutation_rate,
            "fitness_norm": args.fitness_norm,
            **target_entry,
        }

        metrics = metric_tracker.compute(source, adversarial)
        summary.update(metrics)
        final_result.update(ssim=float(metrics["ssim"]), lpips=float(metrics["lpips"]))
        p.dump(final_result, open(final_result_path, 'wb'))

        if args.save_imgs:
            output = adversarial if args.setting == "realistic" else adversarial * 255.0
            Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(
                f"{example_dir}/{summary['adversarial']}_{save_file}.JPEG"
            )

        with open(result_path, "w") as file:
            json.dump(summary, file, indent=4, cls=NumpyEncoder)

        _print_result(path_img, summary)
        successes.append(bool(summary["adversarial"]))
        l2_values.append(float(summary["l2_distance"]))
        areas.append(float(summary["patch_area_ratio"]))
        query_values.append(int(summary["queries"]))

    if not successes:
        print("No eligible images were evaluated.")
        return
    print(f"Average Attack Success Rate: {100 * np.mean(successes):.2f}")
    print(f"L2 (mean, std): {np.mean(l2_values):.2f} ({np.std(l2_values):.2f})")
    print(f"Patch area (mean): {100 * np.mean(areas):.4f}%")
    print(f"Queries (mean): {np.mean(query_values):.2f}")
    metric_tracker.print_summary()


if __name__ == "__main__":
    main()
