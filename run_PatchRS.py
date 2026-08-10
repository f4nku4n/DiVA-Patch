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
    parser.add_argument("--exp_root",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_result"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=40, help="patch size")
    parser.add_argument("--p_init", type=float, default=0.4, help="initial Patch-RS sampling probability")
    parser.add_argument("--update_loc_period", type=int, default=4, help="number of queries between location updates")
    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--dataset", choices=list(VALID_MODELS), default="ImageNet1K")
    parser.add_argument("--vision_model", default="VGGNet16", choices=["VGGNet16", "ResNet50", "ViT16", "EfficientNetV2S", "Swin"])
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--dataset_root", required=True, help="ImageNet1K path")
    parser.add_argument("--attack_type", default="non_targeted", choices=["targeted", "non_targeted"])
    parser.add_argument("--save_imgs", action="store_true", help="save adversarial images")
    parser.add_argument("--demo", action="store_true", help="attack example images")
    return parser


def _components(dataset, vision_model, setting):
    if vision_model not in VALID_MODELS[dataset]:
        allowed = ", ".join(VALID_MODELS[dataset])
        raise ValueError(f"--vision_model {vision_model!r} is not valid for --dataset {dataset!r}; choose one of: {allowed}")
    if setting == "common":
        from attack_methods.PatchRS_common import PatchRS_common as PatchRS
    else:
        from attack_methods.PatchRS import PatchRS
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
    return PatchRS, ModelClass, model_args


def _image_loader(dataset, setting):
    if setting == "realistic":
        def load_realistic(path):
            return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)

        return load_realistic

    from torchvision import transforms

    if dataset == "Flower102":
        resize = transforms.Resize((256, 256))
    elif dataset == "Food101":
        resize = transforms.Resize((224, 224))
    else:
        resize = transforms.Resize(256)
    operations = [resize]
    if dataset != "Food101":
        operations.append(transforms.CenterCrop(224))
    operations.append(transforms.ToTensor())
    transform = transforms.Compose(operations)

    def load_common(path):
        return pytorch_switch(transform(Image.open(path).convert("RGB"))).detach().numpy()

    return load_common


def _save_folder(args):
    name = f"PatchRS-{args.dataset}-{args.vision_model}-{args.setting}-{args.attack_type}/SEED_{args.seed}"
    return os.path.join(args.exp_root, name)


def _label_file(args):
    if args.demo:
        if args.dataset != "ImageNet1K":
            raise ValueError("--demo is only supported with --dataset ImageNet1K")
        return "TEST_IMGs/demo_targeted.json" if args.attack_type == "targeted" else "TEST_IMGs/demo_untargeted.json"
    if args.dataset == "Flower102":
        return "TEST_IMGs/EfficientNetV2S_Flower102.json"
    if args.dataset == "Food101":
        return "TEST_IMGs/Swin_Food101.json"
    return f"TEST_IMGs/{args.vision_model}_ImgNet1K.json"


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num_images must be positive")
    if args.max_query <= 0:
        parser.error("--max_query must be positive")
    if args.patch_size <= 0:
        parser.error("--patch_size must be positive")
    if args.update_loc_period <= 0:
        parser.error("--update_loc_period must be positive")

    try:
        PatchRS, ModelClass, model_args = _components(args.dataset, args.vision_model, args.setting)
        label_file = _label_file(args)
    except ValueError as exc:
        parser.error(str(exc))
    model = ModelClass(*model_args, args.device)
    set_seed(args.seed)

    with open(label_file) as file:
        path_labels = json.load(file)

    save_folder = _save_folder(args)
    process_folder = os.path.join(save_folder, "processes")
    result_folder = os.path.join(save_folder, "results")
    example_folder = os.path.join(save_folder, "examples")
    for folder in (save_folder, process_folder, result_folder, example_folder):
        os.makedirs(folder, exist_ok=True)

    path_labels = sample_image_labels(
        path_labels,
        args.num_images,
        args.seed,
        label_file,
        os.path.join(save_folder, "sampled_images.json")
    )
    metric_tracker = PerceptualMetricTracker(device=args.device, data_range=1.0 if args.setting == "common" else 255.0)
    load_image = _image_loader(args.dataset, args.setting)
    adversarial, l2_values = [], []

    for index, path_img in enumerate(path_labels, start=1):
        save_file = path_img.replace(".JPEG", "").replace("/", "_")
        process_path = os.path.join(process_folder, f"{save_file}.p")
        result_json = os.path.join(result_folder, f"{save_file}.json")
        final_result_path = os.path.join(result_folder, f"{save_file}_result.p")
        if os.path.exists(result_json):
            with open(result_json) as file:
                existing_result = json.load(file)
            if metric_tracker.is_complete(existing_result) and "first_success_query" in existing_result and os.path.exists(final_result_path):
                metric_tracker.add_summary(existing_result)
                metric_tracker.print_result(path_img, existing_result)
                adversarial.append(existing_result["adversarial"])
                l2_values.append(existing_result["l2_distance"])
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
        else:
            if loss.get_label(img_cls) != true_label:
                continue
            set_seed(args.seed)
            attacker = PatchRS(img_cls=img_cls, loss_function=loss, max_query=args.max_query, p_init=args.p_init, patch_size=[args.patch_size, args.patch_size], update_loc_period=args.update_loc_period)
            attacker.run()
            process = attacker.process

        best = process[-1]
        img_adv = img_cls.copy()
        loc_x, loc_y = best[2]
        patch = best[3]
        img_adv[loc_x:loc_x + patch.shape[0], loc_y:loc_y + patch.shape[1], :] = patch
        metrics = metric_tracker.compute(img_cls, img_adv)
        summary = {
            "adversarial": best[1],
            "l2_distance": best[-2],
            "location": best[2],
            "loss": best[-1],
            "first_success_query": first_success_query_from_process(process, success_index=1, query_index=0),
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
        }
        final_result = {
            "adversarial": bool(best[1]),
            "location": best[2],
            "patch": np.asarray(best[3]).copy(),
            "l2_distance": float(best[-2]),
            "loss": float(best[-1]),
            "queries": int(best[0]),
            "first_success_query": summary["first_success_query"],
            "ssim": float(metrics["ssim"]),
            "lpips": float(metrics["lpips"]),
            "setting": args.setting,
            "attack_type": args.attack_type,
            "patch_size": args.patch_size,
            "p_init": args.p_init,
            "update_loc_period": args.update_loc_period,
        }
        if args.save_imgs:
            with open(process_path, "wb") as file:
                pickle.dump(process, file)

        with open(final_result_path, "wb") as file:
            pickle.dump(final_result, file)

        metric_tracker.print_result(path_img, summary)
        with open(result_json, "w") as file:
            json.dump(summary, file, indent=4, cls=NumpyEncoder)

        if args.save_imgs:
            output = img_adv * 255.0 if args.setting == "common" else img_adv
            Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(os.path.join(example_folder, f"{bool(best[1])}_{path_img.replace('/', '_')}"))

        adversarial.append(bool(best[1]))
        l2_values.append(float(best[-2]))

    if not adversarial:
        print("No eligible images were evaluated.")
        return
    print(f"Average Attack Success Rate: {100 * np.mean(adversarial):.2f}")
    print(f"L2 (mean, std): {np.mean(l2_values):.2f} ({np.std(l2_values):.2f})")
    metric_tracker.print_summary()


if __name__ == "__main__":
    main()
