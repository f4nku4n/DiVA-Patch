import argparse
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attack_methods.WhiteBoxPatch import LaVAN, MaskedAutoPGD, MaskedPGD
from models.ImageNetModels import ImageNetModel
from utils import NumpyEncoder, PerceptualMetrics, pytorch_switch, set_seed


ATTACKS = {
    "MaskedPGD": MaskedPGD,
    "MaskedAutoPGD": MaskedAutoPGD,
    "LaVAN": LaVAN,
}

DEFAULTS = {
    "MaskedPGD": {"steps": 100, "eps": 1.0, "step_size": 0.01},
    "MaskedAutoPGD": {"steps": 100, "eps": 0.3, "step_size": 0.1},
    "LaVAN": {"steps": 500, "eps": 1.0, "step_size": 5.0},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack_method",
        choices=list(ATTACKS),
        default="MaskedPGD",
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
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    model_index = {"VGGNet16": 0, "ResNet50": 1, "ViT16": 2}
    model = ImageNetModel(model_index[args.vision_model], args.device)
    perceptual_metrics = None
    load_image = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
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
    save_folder = (
        f"exp_result/{args.attack_method}-{args.vision_model}-SEED_{args.seed}"
        f"-common-{args.attack_type}"
    )
    for directory in ("processes", "results", "examples"):
        os.makedirs(os.path.join(save_folder, directory), exist_ok=True)

    successes, distances, ssim_scores, lpips_scores = [], [], [], []
    for index, path_img in enumerate(path_labels):
        save_file = path_img.replace(".JPEG", "").replace("/", "_")
        result_path = os.path.join(save_folder, "results", save_file + ".json")
        if os.path.exists(result_path):
            continue

        print(f"Image #{index + 1}: {path_img}")
        image_path = os.path.join(args.dataset_root, path_img)
        image_tensor = load_image(Image.open(image_path).convert("RGB"))
        image = pytorch_switch(image_tensor).detach().numpy()
        true_label = path_labels[path_img]["true_label"]
        target_label = path_labels[path_img]["target_label"]

        with torch.inference_mode():
            clean = image_tensor[None, :].to(args.device)
            initial_label = int(model.predict(clean).argmax(dim=1).item())
        if initial_label != true_label:
            continue

        set_seed(args.seed)
        attack = ATTACKS[args.attack_method](
            image=image,
            model=model,
            true_label=true_label,
            target_label=target_label,
            targeted=targeted,
            patch_size=(args.patch_h, args.patch_w),
            steps=steps,
            step_size=step_size,
            eps=eps,
            clip_min=0.0,
            clip_max=1.0,
            device=args.device,
            location_update_period=args.location_update_period,
        )
        result = attack.run()
        if perceptual_metrics is None:
            perceptual_metrics = PerceptualMetrics(device=args.device)
        metrics = perceptual_metrics(image, result["image"], data_range=1.0)

        with open(
            os.path.join(save_folder, "processes", save_file + ".p"), "wb"
        ) as file:
            pickle.dump(result["process"], file)

        summary = {
            "adversarial": result["adversarial"],
            "prediction": result["prediction"],
            "location": result["location"],
            "l2_distance": result["l2"],
            "loss": result["loss"],
            "steps": steps,
            "eps": eps,
            "step_size": step_size,
            "location_update_period": args.location_update_period,
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
        }
        with open(result_path, "w") as file:
            json.dump(summary, file, indent=4, cls=NumpyEncoder)

        output = np.clip(result["image"] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(output).save(
            os.path.join(
                save_folder,
                "examples",
                f"{result['adversarial']}_{path_img.replace('/', '_')}",
            )
        )
        successes.append(result["adversarial"])
        distances.append(result["l2"])
        ssim_scores.append(metrics["ssim"])
        lpips_scores.append(metrics["lpips"])

    if successes:
        asr = np.mean(successes) * 100
        print(f"Average Attack Success Rate: {asr:.2f}")
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
