# DiVA-Patch: Diverse and Versatile Adversarial Patch Attack
Official implementation of the paper: **DiVA-Patch: Diverse and Versatile Adversarial Patch-based Attack via Quality-Diversity Optimization**

## Setup
- Clone this repo
- Install necessary packages
```
$ cd source_code
$ bash install.sh
```
- Download the validation set of the ImageNet1K, Food101, Flower102 datasets.

## Reproducing the results
This repo have already implemented following black-box patch-based attacks:
- [**Patch-RS**](https://github.com/fra31/sparse-rs)
- [**CamoPatch**](https://github.com/phoenixwilliams/CamoPatch)
- [**DevoPatch**](https://github.com/Omenzychen/DevoPatch)
- **DiVA-Patch** (**Ours**)

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
- [**LOAP**](https://arxiv.org/abs/2005.02313), with full or random location optimization

All black-box and white-box runners accept `--exp_root` to select the root
directory for saved results. It defaults to `exp_result` inside the project:
```shell
python run_WhiteBoxPatch.py --setting common --attack_method LOAP --vision_model VGGNet16 --dataset_root "$DATASET_PATH" --exp_root "/path/to/experiments"
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

PatchRS, DevoPatch, and all white-box methods also save a per-image
`results/*_result.p` artifact. It contains the final patch content and
location together with success, L2, loss or fitness, query information,
SSIM, LPIPS, and method-specific hyperparameters. These artifacts and process
files are always saved; `--save_imgs` controls only example image output.

Before executing the scripts below, please set the DATASET_PATH variable in each *.sh file to the path of your ImageNet-1K validation set.

## Acknowledgement
We want to give our thanks to the authors of [Patch-RS](https://ojs.aaai.org/index.php/AAAI/article/view/20595) and [CamoPatch](https://openreview.net/forum?id=B94G0MXWQX) for their valuable reposistories.
