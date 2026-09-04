# DiVA-Patch: Diverse and Versatile Adversarial Patch Attack
Official implementation of the paper: **One Run, Many Threats: Diverse Adversarial Patches via Quality-Diversity Optimization**

## Setup
- Clone this repo
- Install necessary packages
```
$ cd source_code
$ bash install.sh
```
- Download the validation set of the [ImageNet1K](https://www.image-net.org/download.php) dataset.
```
val/
|---n01440764
    |---*.JPEG
|---n01443537
    |---*.JPEG
...
|---n15075141
    |---*.JPEG
```
## Reproducing the results
This repo have already implemented following black-box patch-based attacks:
- [**Patch-RS**](https://github.com/fra31/sparse-rs)
- [**CamoPatch**](https://github.com/phoenixwilliams/CamoPatch)
- [**DevoPatch**](https://arxiv.org/abs/2307.00477)
- **DiVA-Patch** (**Ours**)

Run DevoPatch under the common or realistic setting with:
```shell
python run_DevoPatch.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_PATH" --device cuda --batch_size 8

python run_DevoPatch.py --setting realistic --vision_model VGGNet16 --dataset_root "$DATASET_PATH" --device cuda --attack_type targeted --batch_size 8
```

DevoPatch defaults to population size `10`, initialization rate `0.35`,
integer mutation rate `1`, L0 fitness, and a strict `10,000` query budget.
For an untargeted attack, its target texture is sampled from a different
class in the full model-specific label dictionary. For a targeted attack, it
is sampled from entries whose true label equals the configured target label.
The selected texture is deterministic for the seed and source path and is
recorded in `target_images.json`.
`--batch_size` controls how many independent images are attacked concurrently;
each active image owns one attacker and contributes at most one query to each
victim-model batch.
The old common and realistic entry points remain compatibility wrappers.

CamoPatch common and realistic settings share one runner:
```shell
python run_CamoPatch.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_PATH"

python run_CamoPatch.py --setting realistic --vision_model VGGNet16 --dataset_root "$DATASET_PATH"
```

Patch-RS uses the same unified runner pattern:
```shell
python run_PatchRS.py --setting common \
  --vision_model VGGNet16 --dataset_root "$DATASET_PATH"

python run_PatchRS.py --setting realistic \
  --vision_model VGGNet16 --dataset_root "$DATASET_PATH"
```
Use `--patch_size`, `--p_init`, and `--update_loc_period` to configure
Patch-RS. The old common and realistic entry points remain compatibility
wrappers.

DiVA-Patch also uses one runner for both settings:
```shell
python run_DiVA_Patch.py --setting common --vision_model VGGNet16 --dataset_root "$DATASET_PATH"

python run_DiVA_Patch.py --setting realistic --vision_model VGGNet16 --dataset_root "$DATASET_PATH" --attack_type targeted
```
Use `--batch_size 8` (or another value that fits GPU memory) to attack that
many images concurrently with one batched victim-model evaluation per query
round. The default is `1` for compatibility; CUDA out-of-memory errors require
rerunning with a smaller value. Use `--save_imgs` to save example images. The
old common and realistic entry points remain compatibility wrappers.

The repository also includes the following white-box localized attacks in
`attack_methods/WhiteBoxPatch.py`:
- **MaskedPGD**
- **MaskedAutoPGD**
- [**LaVAN**](https://proceedings.mlr.press/v80/karmon18a.html)
- [**LOAP**](https://arxiv.org/abs/2005.02313), with full or random
  location optimization

Run any white-box attack under the common or realistic setting with:
```shell
python run_WhiteBoxPatch.py --setting common --attack_method MaskedPGD --vision_model VGGNet16 --dataset_root "$DATASET_PATH" --device cuda

python run_WhiteBoxPatch.py --setting realistic --attack_method LaVAN --vision_model VGGNet16 --dataset_root "$DATASET_PATH" --device cuda --attack_type targeted
```

The old `run_WhiteBoxPatch_common.py` and
`run_WhiteBoxPatch_realistic.py` entry points remain compatibility wrappers.

`--attack_method` accepts `MaskedPGD`, `MaskedAutoPGD`, `LaVAN`, or `LOAP`. Both
runners also accept `--steps` (or `--max_query`), `--patch_h`, `--patch_w`,
`--eps`, `--step_size`, and `--location_update_period`. A period of `0`
keeps one fixed location; a positive value samples a new location after that
many optimization steps and carries the best patch to the new location.
MaskedPGD and MaskedAutoPGD default to 100 steps. LaVAN follows the linked
PyTorch reproduction with a random `[0, 1]` patch, 500 steps, and raw-gradient
step size `5.0`.
LOAP defaults to 100 steps, signed-gradient step size `0.05`, full
four-direction location optimization, stride `2`, and one random restart.
Use `--lo_mode random` for RandLO, `--attempts` for additional restarts,
and `--exclude_box TOP LEFT HEIGHT WIDTH` to prevent overlap with a region.

The common white-box runner batches attacks with `--batch_size` (default `8`).
All white-box runners also accept `--early_stop`; when enabled, each image
stops as soon as its accepted attack state succeeds. In a common batch,
unfinished images continue independently.

All black-box and white-box runners accept `--exp_root` to select the root
directory for saved results. It defaults to `exp_result` inside the project:
```shell
python run_WhiteBoxPatch.py --setting common --attack_method LOAP \
  --vision_model VGGNet16 --dataset_root "$DATASET_PATH" \
  --exp_root "/path/to/experiments"
```

All runners sample `--num_images` paths from the selected label dictionary
(default `100`). Sampling is deterministic for a given seed and is recorded in
`sampled_images.json` inside the experiment folder so resumed runs reuse the
same ordered subset. Demo dictionaries smaller than the requested size use all
available images.

Result JSON files for all black-box and white-box attacks report full-image
`ssim` and `lpips`. LPIPS uses the official learned metric with a pretrained
AlexNet backbone. Existing black-box process artifacts are backfilled without
rerunning the attack. Both metrics are also available through
`utils.PerceptualMetrics`.

### Train and evaluate with PatchZero

`run_PatchZero.py` contains the complete prepare/train/evaluate workflow. Its
dataset and stage-1 training losses follow the original PatchZero code; the
adaptation is that training pairs and exact masks are reconstructed from this
repository's saved attack artifacts.

Prepare a dataset from one or more attack runs:

```shell
python run_PatchZero.py --mode prepare \
  --experiment_dir exp_result/PatchRS-VGGNet16-realistic-non_targeted/SEED_42 \
  --experiment_dir exp_result/DevoPatch-VGGNet16-realistic-non_targeted/SEED_42 \
  --dataset_root "$DATASET_PATH" --patchzero_dataset patchzero_dataset \
  --setting realistic
```

Train stage 1. As in the authors' repository, `--patchzero_repo` must contain
the `pspnet` package from `Lextal/pspnet-pytorch`:

```shell
python run_PatchZero.py --mode train --patchzero_dataset patchzero_dataset \
  --patchzero_repo /path/to/PatchZero --backend resnet50 --device cuda \
  --epochs 20 --batch_size 8 --models_path patchzero_checkpoints
```

Evaluate attack success before/after the trained defense:

```shell
python run_PatchZero.py --mode evaluate \
  --experiment_dir exp_result/PatchRS-VGGNet16-realistic-non_targeted/SEED_42 \
  --dataset_root "$DATASET_PATH" \
  --patchzero_repo /path/to/PatchZero \
  --checkpoint /path/to/PSPNet_5 \
  --backend resnet50 --vision_model VGGNet16 --setting realistic \
  --attack_type non_targeted --device cuda --save_images
```

The per-image report is written to `patchzero_results.csv`. Use the same
command for DiVA-Patch, Patch-RS, DevoPatch, CamoPatch, and white-box result
folders. Missing final-patch artifacts are skipped explicitly.

For inference from Python, load the model once and call the defense directly:

```python
from run_PatchZero import PatchZeroDefense

defense = PatchZeroDefense("/path/to/PatchZero", "patchzero_checkpoints/PSPNet_20")
defended_image = defense(adversarial_image)
```

PatchRS, DevoPatch, and all white-box methods also save a per-image
`results/*_result.p` artifact. It contains the final patch content and
location together with success, L2, loss or fitness, query information,
SSIM, LPIPS, and method-specific hyperparameters. These artifacts and process
files are always saved; `--save_imgs` controls only example image output.

Before executing the scripts below, please set the DATASET_PATH variable in each *.sh file to the path of your ImageNet-1K validation set.

### Demo
You can verify the performance of DiVA-Patch and compare it to other black-box patch attacks on a few demo examples
```shell
$ source bash/run_demo_untargeted.sh
$ source bash/run_demo_targeted.sh
$ source bash/demo_attack_under_common_setting.sh
```

### Main results
#### Directed (_untargeted_) attacks under the _realistic_ settings (i.e., attacking on 1000 _raw_ images)
```shell
$ source bash/run_DiVA_Patch_realistic_non_targeted.sh  # for DiVA-Patch (ours)
$ source bash/run_DevoPatch_realistic_non_targeted.sh  # for DevoPatch
$ source bash/run_PatchRS_realistic_non_targeted.sh  # for PatchRS
$ source bash/run_CamoPatch_realistic_non_targeted.sh  # for CamoPatch
```

#### Directed (_targeted_) attacks under the _realistic_ settings (i.e., attacking on 1000 _raw_ images)
```shell
$ source bash/run_DiVA_Patch_realistic_targeted.sh  # for DiVA-Patch (ours)
$ source bash/run_DevoPatch_realistic_targeted.sh  # for DevoPatch
$ source bash/run_PatchRS_realistic_targeted.sh  # for PatchRS
$ source bash/run_CamoPatch_realistic_targeted.sh  # for CamoPatch
```

### Ablation studies
#### Performance on common attack settings (i.e., attacking on 1000 _transformed_ images)
```shell
$ source bash/attack_under_common_setting.sh
```
#### Effect of K-value to performance of DiVA-Patch
```shell
$ source bash/run_ablation_study_K_effect.sh
```
## Acknowledgement
We want to give our thanks to the authors of [Patch-RS](https://ojs.aaai.org/index.php/AAAI/article/view/20595) and [CamoPatch](https://openreview.net/forum?id=B94G0MXWQX) for their valuable reposistories.
