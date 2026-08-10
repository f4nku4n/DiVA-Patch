import os
import json
import pickle
import argparse

import torch
import numpy as np
from PIL import Image

from utils import set_seed, pytorch_switch, sample_image_labels
from utils import NumpyEncoder, PerceptualMetrics, PerceptualMetricTracker


ATTACK_NAMES = ["MaskedPGD", "MaskedAutoPGD", "LaVAN", "LOAP"]
VALID_MODELS = {
    "ImageNet1K": ("VGGNet16", "ResNet50", "ViT16"),
    "Flower102": ("EfficientNetV2S",),
    "Food101": ("Swin",),
}
DEFAULTS = {
    "MaskedPGD": {"steps": 10000, "eps": 1.0, "step_size": 0.01, "location_update_period": 100},
    "MaskedAutoPGD": {"steps": 10000,  "eps": 0.3, "step_size": 0.1, "location_update_period": 100},
    "LaVAN": {"steps": 10000, "eps": 1.0, "step_size": 5.0, "location_update_period": 100},
    "LOAP": {"steps": 100, "attempts": 100, "eps": 1.0, "step_size": 0.05},
}


def build_parser(default_setting=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["common", "realistic"], default=default_setting or "realistic", help="common uses batched 224x224 images; realistic uses raw images one at a time")
    parser.add_argument("--exp_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_result"))

    parser.add_argument("--attack_method", choices=ATTACK_NAMES, default="MaskedPGD")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--steps", "--max_query", dest="steps", type=int)
    parser.add_argument("--dataset", choices=list(VALID_MODELS), default="ImageNet1K")
    parser.add_argument("--vision_model", choices=["VGGNet16", "ResNet50", "ViT16", "EfficientNetV2S", "Swin"], default="VGGNet16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--attack_type", choices=["targeted", "non_targeted"], default="non_targeted")
    parser.add_argument("--patch_h", type=int, default=40)
    parser.add_argument("--patch_w", type=int, default=40)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--step_size", type=float)
    parser.add_argument("--location_update_period", type=int, default=0)
    parser.add_argument("--lo_mode", choices=["full", "random"], default="full")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--exclude_box", type=int, nargs=4)
    parser.add_argument("--batch_size", type=int, default=8, help="used only by the common setting")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--save_imgs", action="store_true", help="save adversarial images")
    parser.add_argument("--demo", action="store_true")
    return parser


def _components(dataset, vision_model, setting):
    if setting == "common":
        from attack_methods.WhiteBoxPatchBatch import BatchedLOAP, BatchedLaVAN, BatchedMaskedAutoPGD, BatchedMaskedPGD
        attacks = {"MaskedPGD": BatchedMaskedPGD, "MaskedAutoPGD": BatchedMaskedAutoPGD, "LaVAN": BatchedLaVAN, "LOAP": BatchedLOAP}
    else:
        from attack_methods.WhiteBoxPatch import LOAP, LaVAN, MaskedAutoPGD, MaskedPGD
        attacks = {"MaskedPGD": MaskedPGD, "MaskedAutoPGD": MaskedAutoPGD, "LaVAN": LaVAN, "LOAP": LOAP}

    if dataset == "ImageNet1K":
        if setting == "common":
            from models.ImageNetModels import ImageNetModel as ModelClass
        else:
            from models.ImageNetModels_realistic import ImageNetModel as ModelClass
        model_args = ({"VGGNet16": 0, "ResNet50": 1, "ViT16": 2}[vision_model],)
    elif dataset == "Flower102":
        if setting == "common":
            from models.Flower102Models import Flower102Model as ModelClass
        else:
            from models.Flower102Models import Flower102ModelRealistic as ModelClass
        model_args = ()
    else:
        if setting == "common":
            from models.Food101Models import Food101Model as ModelClass
        else:
            from models.Food101Models import Food101ModelRealistic as ModelClass
        model_args = ()

    return attacks, ModelClass, model_args


def _resolved_hyperparameters(args):
    defaults = DEFAULTS[args.attack_method]
    steps = defaults["steps"] if args.steps is None else args.steps
    eps = defaults["eps"] if args.eps is None else args.eps
    step_size = defaults["step_size"] if args.step_size is None else args.step_size
    return steps, eps, step_size


def _label_file(args):
    if args.demo:
        if args.dataset != "ImageNet1K":
            raise ValueError("--demo is only supported with --dataset ImageNet1K")
        return f"TEST_IMGs/demo_{'targeted' if args.attack_type == 'targeted' else 'untargeted'}.json"
    if args.dataset == "Flower102":
        return "TEST_IMGs/EfficientNetV2S_Flower102.json"
    if args.dataset == "Food101":
        return "TEST_IMGs/Swin_Food101.json"
    return f"TEST_IMGs/{args.vision_model}_ImgNet1K.json"


def _save_folder(args):
    return f"{args.exp_root}/{args.attack_method}-{args.dataset}-{args.vision_model}-{args.setting}-{args.attack_type}/SEED_{args.seed}"


def _add_existing(path_img, result_path, aggregates):
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
    aggregates["successes"].append(summary["adversarial"])
    aggregates["distances"].append(summary["l2_distance"])
    aggregates["ssim"].append(summary["ssim"])
    aggregates["lpips"].append(summary["lpips"])


def _summary(args, result, metrics, steps, eps, step_size):
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
        "early_stop": args.early_stop,
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
    }
    if args.setting == "common":
        summary["batch_size"] = args.batch_size
    if args.attack_method == "LOAP":
        summary.update(lo_mode=args.lo_mode, stride=args.stride, attempts=args.attempts, exclude_box=args.exclude_box)
    return summary


def _record_result(args, save_folder, entry, result, metrics, summary, aggregates):
    final_result = {
        "adversarial": result["adversarial"],
        "prediction": result["prediction"],
        "location": result["location"],
        "patch": np.asarray(result["patch"]).copy(),
        "l2_distance": result["l2"],
        "loss": result["loss"],
        "queries": result["queries"],
        "first_success_query": result["first_success_query"],
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
        "attack_method": args.attack_method,
        "setting": args.setting,
        "attack_type": args.attack_type,
        "steps": summary["steps"],
        "eps": summary["eps"],
        "step_size": summary["step_size"],
        "location_update_period": summary["location_update_period"],
        "early_stop": summary["early_stop"],
    }
    if args.attack_method == "LOAP":
        final_result.update(lo_mode=args.lo_mode, stride=args.stride, attempts=args.attempts, exclude_box=args.exclude_box)
    if args.save_imgs:
        with open(entry["process_path"], "wb") as file:
            pickle.dump(result["process"], file)
    with open(entry["final_result_path"], "wb") as file:
        pickle.dump(final_result, file)
    with open(entry["result_path"], "w") as file:
        json.dump(summary, file, indent=4, cls=NumpyEncoder)
    if args.save_imgs:
        scale = 255.0 if args.setting == "common" else 1.0
        output = np.clip(result["image"] * scale, 0, 255).astype(np.uint8)
        Image.fromarray(output).save(os.path.join(save_folder, "examples", f"{result['adversarial']}_{entry['path'].replace('/', '_')}"))
    PerceptualMetricTracker.print_result(entry["path"], summary)
    aggregates["successes"].append(result["adversarial"])
    aggregates["distances"].append(result["l2"])
    aggregates["ssim"].append(metrics["ssim"])
    aggregates["lpips"].append(metrics["lpips"])


def _run_common(args, model, attack_class, path_labels, save_folder, steps, eps, step_size, aggregates):
    from torchvision import transforms

    if args.dataset == "Flower102":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    elif args.dataset == "Food101":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    perceptual_metrics = None
    targeted = args.attack_type == "targeted"

    def run_batch(entries):
        nonlocal perceptual_metrics
        if not entries:
            return
        clean = torch.stack([entry["tensor"] for entry in entries]).to(args.device)
        with torch.inference_mode():
            predictions = model.predict(clean).argmax(dim=1).cpu().tolist()
        entries = [entry for entry, prediction in zip(entries, predictions) if prediction == entry["true_label"]]
        if not entries:
            return
        attack_kwargs = dict(images=np.stack([entry["image"] for entry in entries]), model=model, true_labels=[entry["true_label"] for entry in entries], target_labels=[entry["target_label"] for entry in entries], targeted=targeted, patch_size=(args.patch_h, args.patch_w), steps=steps, step_size=step_size, eps=eps, device=args.device, location_update_period=args.location_update_period, early_stop=args.early_stop)
        if args.attack_method == "LOAP":
            attack_kwargs.update(lo_mode=args.lo_mode, stride=args.stride, attempts=args.attempts, exclude_box=args.exclude_box)
        try:
            results = attack_class(**attack_kwargs).run()
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError("CUDA out of memory during batched white-box attack; reduce --batch_size and rerun.") from error
        if perceptual_metrics is None:
            perceptual_metrics = PerceptualMetrics(device=args.device)
        for entry, result in zip(entries, results):
            metrics = perceptual_metrics(entry["image"], result["image"], data_range=1.0)
            summary = _summary(args, result, metrics, steps, eps, step_size)
            _record_result(args, save_folder, entry, result, metrics, summary, aggregates)

    pending = []
    for index, path_img in enumerate(path_labels, start=1):
        save_file = path_img.replace(".JPEG", "").replace("/", "_")
        result_path = os.path.join(save_folder, "results", save_file + ".json")
        final_result_path = os.path.join(save_folder, "results", save_file + "_result.p")
        if os.path.exists(result_path) and os.path.exists(final_result_path):
            _add_existing(path_img, result_path, aggregates)
            continue
        print(f"Image #{index}: {path_img}")
        tensor = transform(Image.open(os.path.join(args.dataset_root, path_img)).convert("RGB"))
        pending.append({"path": path_img, "tensor": tensor, "image": pytorch_switch(tensor).numpy(), "true_label": path_labels[path_img]["true_label"], "target_label": path_labels[path_img]["target_label"], "result_path": result_path, "final_result_path": final_result_path, "process_path": os.path.join(save_folder, "processes", save_file + ".p")})
        if len(pending) == args.batch_size:
            run_batch(pending)
            pending = []
    run_batch(pending)


def _run_realistic(args, model, attack_class, path_labels, save_folder, steps, eps, step_size, aggregates):
    perceptual_metrics = None
    targeted = args.attack_type == "targeted"
    for index, path_img in enumerate(path_labels, start=1):
        save_file = path_img.replace(".JPEG", "").replace("/", "_")
        result_path = os.path.join(save_folder, "results", save_file + ".json")
        final_result_path = os.path.join(save_folder, "results", save_file + "_result.p")
        if os.path.exists(result_path) and os.path.exists(final_result_path):
            _add_existing(path_img, result_path, aggregates)
            continue
        print(f"Image #{index}: {path_img}")
        image = np.asarray(Image.open(os.path.join(args.dataset_root, path_img)).convert("RGB"), dtype=np.float32)
        true_label = path_labels[path_img]["true_label"]
        target_label = path_labels[path_img]["target_label"]
        with torch.inference_mode():
            clean = torch.from_numpy(image).permute(2, 0, 1)[None, :].to(args.device)
            initial_label = int(model.predict(clean).argmax(dim=1).item())
        if initial_label != true_label:
            print('Wrong label fucking shit')
            continue
        set_seed(args.seed)
        attack_kwargs = dict(image=image, model=model, true_label=true_label, target_label=target_label, targeted=targeted, patch_size=(args.patch_h, args.patch_w), steps=steps, step_size=step_size, eps=eps, clip_min=0.0, clip_max=255.0, device=args.device, location_update_period=args.location_update_period, early_stop=args.early_stop)
        if args.attack_method == "LOAP":
            attack_kwargs.update(lo_mode=args.lo_mode, stride=args.stride, attempts=args.attempts, exclude_box=args.exclude_box)
        result = attack_class(**attack_kwargs).run()
        if perceptual_metrics is None:
            perceptual_metrics = PerceptualMetrics(device=args.device)
        metrics = perceptual_metrics(image, result["image"], data_range=255.0)
        entry = {"path": path_img, "result_path": result_path, "final_result_path": final_result_path, "process_path": os.path.join(save_folder, "processes", save_file + ".p")}
        summary = _summary(args, result, metrics, steps, eps, step_size)
        _record_result(args, save_folder, entry, result, metrics, summary, aggregates)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.num_images <= 0:
        parser.error("--num_images must be positive")
    if args.patch_h <= 0 or args.patch_w <= 0:
        parser.error("--patch_h and --patch_w must be positive")

    set_seed(args.seed)

    attacks, ModelClass, model_args = _components(args.dataset, args.vision_model, args.setting)
    label_file = _label_file(args)
    model = ModelClass(*model_args, args.device)
    with open(label_file) as file:
        path_labels = json.load(file)

    save_folder = _save_folder(args)
    for directory in ("processes", "results", "examples"):
        os.makedirs(os.path.join(save_folder, directory), exist_ok=True)

    path_labels = sample_image_labels(path_labels, args.num_images, args.seed, label_file, os.path.join(save_folder, "sampled_images.json"))
    steps, eps, step_size = _resolved_hyperparameters(args)
    aggregates = {"successes": [], "distances": [], "ssim": [], "lpips": []}

    runner = _run_common if args.setting == "common" else _run_realistic
    runner(args, model, attacks[args.attack_method], path_labels, save_folder, steps, eps, step_size, aggregates)

    if aggregates["successes"]:
        print(f"Average Attack Success Rate: {np.mean(aggregates['successes']) * 100:.2f}")
        print(f"L2 (mean, std): {np.mean(aggregates['distances']):.2f} ({np.std(aggregates['distances']):.2f})")
        print(f"SSIM (mean): {np.mean(aggregates['ssim']):.4f}")
        print(f"LPIPS-AlexNet (mean): {np.mean(aggregates['lpips']):.4f}")
    else:
        print("No correctly classified, unfinished images were attacked.")


if __name__ == "__main__":
    main()
