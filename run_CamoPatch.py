import argparse
import json
import os

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


def build_parser(default_setting=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["common", "realistic"], default=default_setting or "realistic", help="common resizes/crops to 224 in [0,1]; realistic uses raw images (pre-processing)")
    parser.add_argument("--exp_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_result"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=100, help="number of semi-transparent circles")
    parser.add_argument("--patch_size", type=int, default=40, help="patch size")
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
    if setting == "common":
        from attack_methods.CamoPatch_common import CamoPatch_common as CamoPatch
    else:
        from attack_methods.CamoPatch import CamoPatch
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
    return CamoPatch, ModelClass, model_args


def _image_loader(dataset, setting):
    if setting == "realistic":
        def load_realistic(path):
            image = Image.open(path).convert("RGB")
            return np.asarray(image, dtype=np.float32)

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
        image = Image.open(path).convert("RGB")
        return pytorch_switch(transform(image)).detach().numpy()

    return load_common


def _save_folder(args):
    name = f"CamoPatch-{args.dataset}-{args.vision_model}-{args.setting}-{args.attack_type}/SEED_{args.seed}"
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


def main(default_setting=None):
    parser = build_parser(default_setting)
    args = parser.parse_args()

    try:
        CamoPatch, ModelClass, model_args = _components(args.dataset, args.vision_model, args.setting)
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
        os.path.join(save_folder, "sampled_images.json"),
    )
    metric_tracker = PerceptualMetricTracker(
        device=args.device,
        data_range=1.0 if args.setting == "common" else 255.0,
    )
    load_image = _image_loader(args.dataset, args.setting)
    adversarial, l2_values = [], []

    patch_size = args.patch_size
    for index, path_img in enumerate(path_labels, start=1):
        save_file = path_img.replace(".JPEG", "").replace("/", "_")
        process_path = os.path.join(process_folder, f"{save_file}.npy")
        result_json = os.path.join(result_folder, f"{save_file}.json")
        if os.path.exists(result_json):
            with open(result_json) as file:
                existing_result = json.load(file)
            if metric_tracker.is_complete(existing_result) and "first_success_query" in existing_result:
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

        if not os.path.exists(process_path):
            if loss.get_label(img_cls) != true_label:
                continue
            params = {
                "x": img_cls,
                "eps": patch_size**2,
                "n_queries": args.max_query,
                "save_directory": process_path,
                "c": img_cls.shape[2],
                "h": img_cls.shape[0],
                "w": img_cls.shape[1],
                "N": args.N,
                "update_loc_period": 4,
                "mut": 0.3,
                "temp": 300,
            }
            set_seed(args.seed)
            attacker = CamoPatch(params, loss, args.max_query)
            attacker.run()

        artifact = np.load(process_path, allow_pickle=True).item()
        process = artifact["process"]
        success = bool(artifact["adversarial"])
        location = artifact["loc"]
        patch = artifact["patch"]
        l2_distance = process[-1][-2]
        final_loss = process[-1][-1]

        img_adv = img_cls.copy()
        loc_x, loc_y = location
        img_adv[loc_x:loc_x + patch_size, loc_y:loc_y + patch_size, :] = patch

        metrics = metric_tracker.compute(img_cls, img_adv)
        summary = {
            "adversarial": success,
            "l2_distance": l2_distance,
            "location": location,
            "loss": final_loss,
            "first_success_query": artifact.get(
                "first_success_query",
                first_success_query_from_process(process, success_index=0),
            ),
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
        }
        metric_tracker.print_result(path_img, summary)
        with open(result_json, "w") as file:
            json.dump(summary, file, indent=4, cls=NumpyEncoder)

        if args.save_imgs:
            output = img_adv * 255.0 if args.setting == "common" else img_adv
            Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(
                os.path.join(example_folder, f"{success}_{path_img.replace('/', '_')}")
            )

        adversarial.append(success)
        l2_values.append(l2_distance)

    if not adversarial:
        print("No eligible images were evaluated.")
        return
    print(f"Average Attack Success Rate: {100 * np.mean(adversarial):.2f}")
    print(f"L2 (mean, std): {np.mean(l2_values):.2f} ({np.std(l2_values):.2f})")
    metric_tracker.print_summary()


if __name__ == "__main__":
    main()
