import os
import json
import argparse
import pickle as p
from dataclasses import dataclass

import numpy as np
from PIL import Image

from utils import BatchEvaluator, NumpyEncoder, PerceptualMetricTracker
from utils import first_success_query_from_process, sample_image_labels, set_seed

from attack_methods.PatchRS import PatchRS
from attack_methods.controller import AttackController, AttackJob

from utils.LossFunctions import Targeted, UnTargeted, _torch_input
from factory import getVisionModel, getImageLoader, getLabelFile

VALID_MODELS = {
    "ImageNet1K": ("VGGNet16", "ResNet50", "ViT16"),
    "Flower102": ("EfficientNetV2S",),
    "Food101": ("Swin",),
}

@dataclass
class PatchRSContext:
    path_img: str; save_file: str; image: np.ndarray; true_label: int; target_label: int
    process_path: str; result_json: str; final_result_path: str


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["common", "realistic"],
                        default="realistic",
                        help="common resizes/crops to 224 in [0,1]; realistic uses raw images (pre-processing)")
    parser.add_argument("--exp_root", default="./exp_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=40, help="patch size")
    parser.add_argument("--p_init", type=float, default=0.4, help="initial Patch-RS sampling probability")
    parser.add_argument("--update_loc_period", type=int, default=4, help="number of queries between location updates")
    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", choices=list(VALID_MODELS), default="ImageNet1K")
    parser.add_argument("--vision_model", default="VGGNet16", choices=["VGGNet16", "ResNet50", "ViT16", "EfficientNetV2S", "Swin"])
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--dataset_root", required=True, help="ImageNet1K path")
    parser.add_argument("--attack_type", default="non_targeted", choices=["targeted", "non_targeted"])
    parser.add_argument("--save_imgs", action="store_true", help="save adversarial images")
    parser.add_argument("--demo", action="store_true", help="attack example images")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size < 1: parser.error("--batch_size must be at least 1")

    # Load vision model
    VisionModel, model_args = getVisionModel(args.dataset, args.vision_model)
    model = VisionModel(*model_args, args.device)
    evaluator = BatchEvaluator(model)

    label_file = getLabelFile(args)
    path_labels = json.load(open(label_file))

    # Create results folders
    experiment = f"PatchRS-{args.dataset}-{args.vision_model}-{args.setting}-{args.attack_type}"
    save_dir = f'{args.exp_root}/{experiment}/SEED_{args.seed}'

    result_dir = f'{save_dir}/results'
    example_dir = f'{save_dir}/examples'
    process_dir = f'{save_dir}/processes'
    for directory in (save_dir, process_dir, result_dir, example_dir):
        os.makedirs(directory, exist_ok=True)

    path_labels = sample_image_labels(path_labels, args.num_images, args.seed, label_file,
                                      f'{save_dir}/sampled_images.json')
    metric_tracker = PerceptualMetricTracker(device=args.device, data_range=1.0 if args.setting == "common" else 255.0)

    load_image = getImageLoader(args.dataset, args.setting)
    set_seed(args.seed)

    # Run attack
    adversarial, l2_values = [], []

    def finish_batched(job):
        context, process = job.context, job.attacker.process
        best = process[-1]
        img_adv = context.image.copy()
        loc_x, loc_y = best[2]
        patch = np.asarray(best[3])
        img_adv[loc_x:loc_x + patch.shape[0], loc_y:loc_y + patch.shape[1], :] = patch
        metrics = metric_tracker.compute(context.image, img_adv)
        first_success = first_success_query_from_process(process, success_index=1, query_index=0)
        summary = {"adversarial": bool(best[1]), "l2_distance": float(best[-2]),
                   "location": best[2], "loss": float(best[-1]),
                   "first_success_query": first_success, **metrics}
        final_result = {**summary, "patch": patch.copy(), "queries": int(best[0]),
                        "setting": args.setting, "attack_type": args.attack_type,
                        "patch_size": args.patch_size, "p_init": args.p_init,
                        "update_loc_period": args.update_loc_period}
        if args.save_imgs:
            p.dump(process, open(context.process_path, 'wb'))
            output = img_adv * 255.0 if args.setting == "common" else img_adv
            Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(
                f"{example_dir}/{bool(best[1])}_{context.path_img.replace('/', '_')}")
        p.dump(final_result, open(context.final_result_path, 'wb'))
        json.dump(summary, open(context.result_json, 'w'), indent=4, cls=NumpyEncoder)
        metric_tracker.print_result(context.path_img, summary)
        adversarial.append(bool(best[1])); l2_values.append(float(best[-2]))

    if args.batch_size > 1:
        contexts = []
        def run_chunk(chunk):
            labels = evaluator.predict_labels([_torch_input(c.image, False, args.device) for c in chunk])
            jobs = []
            for context, label in zip(chunk, labels):
                if label != context.true_label: continue
                loss = (UnTargeted(model, context.true_label, to_pytorch=True, device=args.device)
                        if args.attack_type == "non_targeted" else
                        Targeted(model, context.true_label, context.target_label, to_pytorch=True, device=args.device))
                attacker = PatchRS(context.image, loss, p_init=args.p_init,
                                   patch_size=[args.patch_size, args.patch_size],
                                   update_loc_period=args.update_loc_period, max_query=args.max_query,
                                   setting=args.setting, rng=np.random.RandomState(args.seed))
                jobs.append(AttackJob(attacker, context))
            AttackController(evaluator, batch_size=args.batch_size, on_complete=finish_batched,
                             description="PatchRS").run(jobs)
        for index, path_img in enumerate(path_labels, start=1):
            save_file = path_img.replace(".JPEG", "").replace("/", "_")
            context = PatchRSContext(
                path_img, save_file, load_image(f'{args.dataset_root}/{path_img}'),
                int(path_labels[path_img]['true_label']), int(path_labels[path_img]['target_label']),
                f'{process_dir}/{save_file}.p', f'{result_dir}/{save_file}.json',
                f'{result_dir}/{save_file}_result.p')
            if os.path.exists(context.result_json) and os.path.exists(context.final_result_path):
                existing = json.load(open(context.result_json))
                if metric_tracker.is_complete(existing) and "first_success_query" in existing:
                    metric_tracker.add_summary(existing); metric_tracker.print_result(path_img, existing)
                    adversarial.append(bool(existing["adversarial"])); l2_values.append(float(existing["l2_distance"])); continue
            if context.image.shape[-1] == 3:
                contexts.append(context)
                if len(contexts) == args.batch_size:
                    run_chunk(contexts); contexts = []
        if contexts: run_chunk(contexts)
        if not adversarial:
            print("No eligible images were evaluated."); return
        print(f"Average Attack Success Rate: {100 * np.mean(adversarial):.2f}")
        print(f"L2 (mean, std): {np.mean(l2_values):.2f} ({np.std(l2_values):.2f})")
        metric_tracker.print_summary(); return

    for index, path_img in enumerate(path_labels, start=1):
        save_file = path_img.replace(".JPEG", "").replace("/", "_")
        process_path = f'{process_dir}/{save_file}.p'
        result_json = f'{result_dir}/{save_file}.json'
        final_result_path = f'{result_dir}/{save_file}_result.p'
        if os.path.exists(result_json):
            existing_result = json.load(open(result_json))
            if metric_tracker.is_complete(existing_result) and "first_success_query" in existing_result and os.path.exists(final_result_path):
                metric_tracker.add_summary(existing_result)
                metric_tracker.print_result(path_img, existing_result)
                adversarial.append(existing_result["adversarial"])
                l2_values.append(existing_result["l2_distance"])
                continue

        print(f'Image #{index}: {path_img}')
        image_path = f'{args.dataset_root}/{path_img}'
        true_label = path_labels[path_img]['true_label']
        target_label = path_labels[path_img]['target_label']
        if args.attack_type == "non_targeted":
            loss = UnTargeted(model, true_label, to_pytorch=True, device=args.device)
        else:
            loss = Targeted(model, true_label, target_label, to_pytorch=True, device=args.device)

        img_cls = load_image(image_path)
        if img_cls.shape[-1] != 3:
            continue

        if os.path.exists(process_path):
            process = p.load(open(process_path, 'rb'))
        else:
            if loss.get_label(img_cls) != true_label:
                continue
            set_seed(args.seed)
            attacker = PatchRS(img_cls=img_cls, loss_function=loss, max_query=args.max_query,
                               p_init=args.p_init, patch_size=[args.patch_size, args.patch_size],
                               update_loc_period=args.update_loc_period, setting=args.setting)
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
            p.dump(process, open(process_path, 'wb'))

        p.dump(final_result, open(final_result_path, 'wb'))

        metric_tracker.print_result(path_img, summary)
        json.dump(summary, open(result_json, 'w'), indent=4, cls=NumpyEncoder)

        if args.save_imgs:
            output = img_adv * 255.0 if args.setting == "common" else img_adv
            Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(
                f"{example_dir}/{bool(best[1])}_{path_img.replace('/', '_')}"
            )

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
