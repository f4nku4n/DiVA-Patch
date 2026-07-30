import argparse
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attack_methods.WhiteBoxPatchBatch import (
    BatchedLOAP,
    BatchedLaVAN,
    BatchedMaskedAutoPGD,
    BatchedMaskedPGD,
)
from models.ImageNetModels import ImageNetModel
from utils import (
    NumpyEncoder,
    PerceptualMetrics,
    PerceptualMetricTracker,
    pytorch_switch,
    sample_image_labels,
    set_seed,
)


ATTACKS = {
    "MaskedPGD": BatchedMaskedPGD,
    "MaskedAutoPGD": BatchedMaskedAutoPGD,
    "LaVAN": BatchedLaVAN,
    "LOAP": BatchedLOAP,
}

DEFAULTS = {
    "MaskedPGD": {"steps": 100, "eps": 1.0, "step_size": 0.01},
    "MaskedAutoPGD": {"steps": 100, "eps": 0.3, "step_size": 0.1},
    "LaVAN": {"steps": 500, "eps": 1.0, "step_size": 5.0},
    "LOAP": {"steps": 100, "eps": 1.0, "step_size": 0.05},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_root",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "exp_result"
        ),
    )
    parser.add_argument(
        "--attack_method", choices=list(ATTACKS), default="MaskedPGD"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", "--max_query", dest="steps", type=int)
    parser.add_argument(
        "--vision_model",
        choices=["VGGNet16", "ResNet50", "ViT16"],
        default="VGGNet16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument(
        "--attack_type",
        choices=["targeted", "non_targeted"],
        default="non_targeted",
    )
    parser.add_argument("--patch_h", type=int, default=40)
    parser.add_argument("--patch_w", type=int, default=40)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--step_size", type=float)
    parser.add_argument("--location_update_period", type=int, default=0)
    parser.add_argument("--lo_mode", choices=["full", "random"], default="full")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--exclude_box", type=int, nargs=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.num_images <= 0:
        parser.error("--num_images must be positive")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    model_index = {"VGGNet16": 0, "ResNet50": 1, "ViT16": 2}
    model = ImageNetModel(model_index[args.vision_model], args.device)
    load_image = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    label_file = (
        f"TEST_IMGs/demo_{'targeted' if args.attack_type == 'targeted' else 'untargeted'}.json"
        if args.demo
        else f"TEST_IMGs/{args.vision_model}_ImgNet1K.json"
    )
    with open(label_file) as file:
        path_labels = json.load(file)

    defaults = DEFAULTS[args.attack_method]
    steps = defaults["steps"] if args.steps is None else args.steps
    eps = defaults["eps"] if args.eps is None else args.eps
    step_size = (
        defaults["step_size"] if args.step_size is None else args.step_size
    )
    targeted = args.attack_type == "targeted"
    save_folder = os.path.join(
        args.exp_root,
        f"{args.attack_method}-{args.vision_model}-SEED_{args.seed}"
        f"-common-{args.attack_type}",
    )
    for directory in ("processes", "results", "examples"):
        os.makedirs(os.path.join(save_folder, directory), exist_ok=True)
    path_labels = sample_image_labels(
        path_labels,
        args.num_images,
        args.seed,
        label_file,
        os.path.join(save_folder, "sampled_images.json"),
    )

    successes, distances, ssim_scores, lpips_scores = [], [], [], []
    perceptual_metrics = None

    def add_existing(path_img, result_path):
        with open(result_path) as file:
            summary = json.load(file)
        changed = False
        for field in ("queries", "first_success_query"):
            if field not in summary:
                summary[field] = None
                changed = True
        if changed:
            with open(result_path, "w") as file:
                json.dump(summary, file, indent=4, cls=NumpyEncoder)
        PerceptualMetricTracker.print_result(path_img, summary)
        successes.append(summary["adversarial"])
        distances.append(summary["l2_distance"])
        ssim_scores.append(summary["ssim"])
        lpips_scores.append(summary["lpips"])

    def run_batch(entries):
        nonlocal perceptual_metrics
        if not entries:
            return
        clean = torch.stack([entry["tensor"] for entry in entries]).to(
            args.device
        )
        with torch.inference_mode():
            predictions = model.predict(clean).argmax(dim=1).cpu().tolist()
        entries = [
            entry
            for entry, prediction in zip(entries, predictions)
            if prediction == entry["true_label"]
        ]
        if not entries:
            return

        attack_kwargs = dict(
            images=np.stack([entry["image"] for entry in entries]),
            model=model,
            true_labels=[entry["true_label"] for entry in entries],
            target_labels=[entry["target_label"] for entry in entries],
            targeted=targeted,
            patch_size=(args.patch_h, args.patch_w),
            steps=steps,
            step_size=step_size,
            eps=eps,
            device=args.device,
            location_update_period=args.location_update_period,
            early_stop=args.early_stop,
        )
        if args.attack_method == "LOAP":
            attack_kwargs.update(
                lo_mode=args.lo_mode,
                stride=args.stride,
                attempts=args.attempts,
                exclude_box=args.exclude_box,
            )
        try:
            results = ATTACKS[args.attack_method](**attack_kwargs).run()
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "CUDA out of memory during batched white-box attack; "
                "reduce --batch_size and rerun."
            ) from error

        if perceptual_metrics is None:
            perceptual_metrics = PerceptualMetrics(device=args.device)
        for entry, result in zip(entries, results):
            metrics = perceptual_metrics(
                entry["image"], result["image"], data_range=1.0
            )
            with open(entry["process_path"], "wb") as file:
                pickle.dump(result["process"], file)
            summary = {
                "adversarial": result["adversarial"],
                "prediction": result["prediction"],
                "location": result["location"],
                "l2_distance": result["l2"],
                "loss": result["loss"],
                "queries": result["queries"],
                "first_success_query": result["first_success_query"],
                "steps": steps,
                "eps": eps,
                "step_size": step_size,
                "location_update_period": args.location_update_period,
                "batch_size": args.batch_size,
                "early_stop": args.early_stop,
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
            }
            if args.attack_method == "LOAP":
                summary.update(
                    lo_mode=args.lo_mode,
                    stride=args.stride,
                    attempts=args.attempts,
                    exclude_box=args.exclude_box,
                )
            with open(entry["result_path"], "w") as file:
                json.dump(summary, file, indent=4, cls=NumpyEncoder)
            output = np.clip(result["image"] * 255.0, 0, 255).astype(
                np.uint8
            )
            Image.fromarray(output).save(
                os.path.join(
                    save_folder,
                    "examples",
                    f"{result['adversarial']}_{entry['path'].replace('/', '_')}",
                )
            )
            PerceptualMetricTracker.print_result(entry["path"], summary)
            successes.append(result["adversarial"])
            distances.append(result["l2"])
            ssim_scores.append(metrics["ssim"])
            lpips_scores.append(metrics["lpips"])

    pending = []
    for index, path_img in enumerate(path_labels):
        save_file = path_img.replace(".JPEG", "").replace("/", "_")
        result_path = os.path.join(save_folder, "results", save_file + ".json")
        if os.path.exists(result_path):
            add_existing(path_img, result_path)
            continue
        print(f"Image #{index + 1}: {path_img}")
        tensor = load_image(
            Image.open(os.path.join(args.dataset_root, path_img)).convert("RGB")
        )
        pending.append(
            {
                "path": path_img,
                "tensor": tensor,
                "image": pytorch_switch(tensor).numpy(),
                "true_label": path_labels[path_img]["true_label"],
                "target_label": path_labels[path_img]["target_label"],
                "result_path": result_path,
                "process_path": os.path.join(
                    save_folder, "processes", save_file + ".p"
                ),
            }
        )
        if len(pending) == args.batch_size:
            run_batch(pending)
            pending = []
    run_batch(pending)

    if successes:
        print(f"Average Attack Success Rate: {np.mean(successes) * 100:.2f}")
        print(
            f"L2 (mean, std): {np.mean(distances):.2f} "
            f"({np.std(distances):.2f})"
        )
        print(f"SSIM (mean): {np.mean(ssim_scores):.4f}")
        print(f"LPIPS-AlexNet (mean): {np.mean(lpips_scores):.4f}")
    else:
        print("No correctly classified, unfinished images were attacked.")


if __name__ == "__main__":
    main()
