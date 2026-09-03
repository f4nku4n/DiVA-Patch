import os
import json
import argparse
import pickle as p
from dataclasses import dataclass

import numpy as np
from PIL import Image

from utils import BatchEvaluator, NumpyEncoder, PerceptualMetricTracker
from utils import first_success_query_from_process, sample_image_labels, set_seed, apply_patch

from attack_methods.DiVA_Patch import DiVA_Patch
from attack_methods.controller import AttackController, AttackJob

from utils.LossFunctions import Targeted, UnTargeted, _torch_input
from factory import getVisionModel, getImageLoader, getLabelFile

VALID_MODELS = {
    "ImageNet1K": ("VGGNet16", "ResNet50", "ViT16"),
    "Flower102": ("EfficientNetV2S",),
    "Food101": ("Swin",),
}


@dataclass
class DiVAJobContext:
    index: int
    path_img: str
    save_file: str
    result_json: str
    process_path: str
    img_cls: np.ndarray
    true_label: int
    target_label: int


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["common", "realistic"],
                        default="realistic",
                        help="common resizes/crops to 224 in [0,1]; realistic uses raw images (pre-processing)")
    parser.add_argument("--exp_root", default='./exp_results')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_query", type=int, default=10000)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="maximum number of images attacked in each evaluator batch")
    parser.add_argument("--dataset", choices=list(VALID_MODELS), default="ImageNet1K")
    parser.add_argument("--vision_model", default="VGGNet16", choices=["VGGNet16", "ResNet50", "ViT16", "EfficientNetV2S", "Swin"])
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--dataset_root", required=True, help="dataset path")
    parser.add_argument("--attack_type", default="non_targeted", choices=["targeted", "non_targeted"])
    parser.add_argument("--N", type=int, default=100, help="number of semi-transparent circles")
    parser.add_argument("--patch_h", type=int, default=40, help="height of patch")
    parser.add_argument("--patch_w", type=int, default=40, help="width of patch")
    parser.add_argument("--grid_w", type=int, default=40, help="width of grid archive")
    parser.add_argument("--grid_h", type=int, default=40, help="height of grid archive")
    parser.add_argument("--delta_max", type=float, default=20.0, help="maximum selection temperature")
    parser.add_argument("--delta_min", type=float, default=0.1, help="minimum selection temperature")
    parser.add_argument("--K", type=int, default=100, help="number of refinement queries")
    parser.add_argument("--save_imgs", action="store_true", help="save adversarial images")
    parser.add_argument("--demo", action="store_true", help="attack example images")
    return parser


def _safe_name(path):
    return path.replace(".JPEG", "").replace("/", "_").replace("\\", "_")

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")

    # Load vision model
    VisionModel, model_args = getVisionModel(args.dataset, args.vision_model)
    model = VisionModel(*model_args, args.device)
    evaluator = BatchEvaluator(model)

    label_file = getLabelFile(args)
    path_labels = json.load(open(label_file))

    # Create results folders
    experiment = f"DiVA_Patch-GridSize_{args.grid_h}_{args.grid_w}-Delta_{args.delta_max}_{args.delta_min}-K{args.K}-{args.dataset}-{args.vision_model}-{args.setting}-{args.attack_type}"
    save_dir = f'{args.exp_root}/{experiment}/SEED_{args.seed}'

    result_dir = f'{save_dir}/results'
    example_dir = f'{save_dir}/examples'
    process_dir = f'{save_dir}/processes'
    map_elites_dir = f'{save_dir}/map_elites'
    for directory in (save_dir, map_elites_dir, process_dir, result_dir, example_dir):
        os.makedirs(directory, exist_ok=True)

    path_labels = sample_image_labels(path_labels, args.num_images, args.seed, label_file, f'{save_dir}/sampled_images.json')
    metric_tracker = PerceptualMetricTracker(device=args.device, data_range=1.0 if args.setting == "common" else 255.0)

    load_image = getImageLoader(args.dataset, args.setting)
    set_seed(args.seed)

    # Run attacks. The source lazily preflights clean labels in batches and only
    # keeps base images for the active attack window on the GPU.
    adversarial_results, l2_values, location_counts = [], [], []

    def build_attack_jobs(contexts):
        clean_inputs = [
            _torch_input(context.img_cls, False, args.device)
            for context in contexts
        ]
        predicted_labels = evaluator.predict_labels(clean_inputs)
        del clean_inputs

        for context, predicted_label in zip(contexts, predicted_labels):
            if predicted_label != context.true_label:
                continue
            if args.attack_type == "non_targeted":
                loss = UnTargeted(
                    model, context.true_label, to_pytorch=True, device=args.device
                )
            else:
                loss = Targeted(
                    model,
                    context.true_label,
                    context.target_label,
                    to_pytorch=True,
                    device=args.device,
                )

            attacker = DiVA_Patch(
                img_cls=context.img_cls,
                loss_function=loss,
                delta_max=args.delta_max,
                delta_min=args.delta_min,
                K=args.K,
                patch_size=[args.patch_h, args.patch_w],
                grid_size=[args.grid_h, args.grid_w],
                max_query=args.max_query,
                setting=args.setting,
                rng=np.random.RandomState(args.seed),
            )
            yield AttackJob(attacker=attacker, context=context)

    def attack_jobs():
        clean_batch = []
        for index, path_img in enumerate(path_labels, start=1):
            save_file = _safe_name(path_img)
            result_json = f'{result_dir}/{save_file}.json'
            process_path = f'{process_dir}/{save_file}_process.p'
            if os.path.exists(result_json):
                existing_result = json.load(open(result_json))
                if metric_tracker.is_complete(existing_result) and "first_success_query" in existing_result:
                    metric_tracker.add_summary(existing_result)
                    metric_tracker.print_result(path_img, existing_result)
                    adversarial_results.append(existing_result["adversarial"])
                    l2_values.append(existing_result["l2_distance"])
                    if existing_result["adversarial"]:
                        location_counts.append(len(existing_result.get("#locs", [])))
                    continue

            print(f'Image #{index}: {path_img}')
            image_path = f'{args.dataset_root}/{path_img}'
            true_label = path_labels[path_img]['true_label']
            target_label = path_labels[path_img]['target_label']
            img_cls = load_image(image_path)
            if img_cls.shape[-1] != 3:
                continue

            # Preserve the existing completed-process recovery path.
            if os.path.exists(process_path):
                process = p.load(open(process_path, 'rb'))
                best = process[-1]
                img_adv = apply_patch(img_cls, best[2], np.asarray(best[3]))
                metrics = metric_tracker.compute(img_cls, img_adv)
                summary = {
                    "adversarial": best[1],
                    "l2_distance": best[4],
                    "first_success_query": first_success_query_from_process(
                        process, success_index=1, query_index=0
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
                        os.path.join(example_dir, f"{best[1]}_{path_img.replace('/', '_')}")
                    )

                adversarial_results.append(best[1])
                l2_values.append(best[4])
                continue

            clean_batch.append(DiVAJobContext(
                index=index,
                path_img=path_img,
                save_file=save_file,
                result_json=result_json,
                process_path=process_path,
                img_cls=img_cls,
                true_label=true_label,
                target_label=target_label,
            ))
            if len(clean_batch) == args.batch_size:
                yield from build_attack_jobs(clean_batch)
                clean_batch = []

        if clean_batch:
            yield from build_attack_jobs(clean_batch)

    def save_completed_job(job):
        attacker = job.attacker
        context = job.context

        map_elites = attacker.return_map_elites()
        p.dump(
            map_elites,
            open(os.path.join(map_elites_dir, f'{context.save_file}_map_elites.p'), 'wb'),
        )

        if args.save_imgs:
            p.dump(attacker.process, open(context.process_path, 'wb'))

        results = []
        for cell_index, individual in attacker.map_elites.grid_final.items():
            if individual is not None and individual.success_attack:
                results.append([
                    cell_index,
                    individual.success_attack,
                    individual.location,
                    individual.patch,
                    individual.l2,
                    individual.loss,
                ])
        p.dump(
            results,
            open(os.path.join(result_dir, f'{context.save_file}_result.p'), 'wb'),
        )

        best_idv = attacker.get_best_quality_solution()
        print(f"Best patch:\n+ Adversarial: {best_idv.success_attack}\n+ L2: {best_idv.l2:.2f}")
        if best_idv.success_attack:
            location_counts.append(len(results))

        img_adv = apply_patch(context.img_cls, best_idv.location, best_idv.patch)
        metrics = metric_tracker.compute(context.img_cls, img_adv)
        summary = {
            "adversarial": best_idv.success_attack,
            "l2_distance": best_idv.l2,
            "#locs": [[result[2], result[4], result[5]] for result in results],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
            "first_success_query": attacker.first_success_query,
        }
        metric_tracker.print_result(context.path_img, summary)
        with open(context.result_json, "w") as file:
            json.dump(summary, file, indent=4, cls=NumpyEncoder)

        if args.save_imgs:
            output = img_adv * 255.0 if args.setting == "common" else img_adv
            Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(
                f"{example_dir}/{best_idv.success_attack}_{context.path_img.replace('/', '_')}"
            )
        l2_values.append(best_idv.l2)
        adversarial_results.append(best_idv.success_attack)

    controller = AttackController(
        evaluator=evaluator,
        batch_size=args.batch_size,
        on_complete=save_completed_job,
        show_progress=True,
    )
    controller.run(attack_jobs())

    if not adversarial_results:
        print("No eligible images were evaluated.")
        return
    print(f"Average Attack Success Rate: {100 * np.mean(adversarial_results):.2f}")
    print(f"L2 (mean, std): {np.mean(l2_values):.2f} ({np.std(l2_values):.2f})")
    if location_counts:
        print(f"#Locs (mean, std): {np.mean(location_counts):.2f} ({np.std(location_counts):.2f})")
    metric_tracker.print_summary()


if __name__ == "__main__":
    main()
