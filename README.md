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
- **DiVA-Patch** (**Ours**)

The repository also includes the following white-box localized attacks in
`attack_methods/WhiteBoxPatch.py`:
- **MaskedPGD**
- **MaskedAutoPGD**
- [**LaVAN**](https://proceedings.mlr.press/v80/karmon18a.html)

Run any white-box attack under the common or realistic setting with:
```shell
python run_WhiteBoxPatch_common.py --attack_method MaskedPGD \
  --vision_model VGGNet16 --dataset_root "$DATASET_PATH" --device cuda

python run_WhiteBoxPatch_realistic.py --attack_method LaVAN \
  --vision_model VGGNet16 --dataset_root "$DATASET_PATH" --device cuda \
  --attack_type targeted
```

`--attack_method` accepts `MaskedPGD`, `MaskedAutoPGD`, or `LaVAN`. Both
runners also accept `--steps` (or `--max_query`), `--patch_h`, `--patch_w`,
`--eps`, `--step_size`, and `--location_update_period`. A period of `0`
keeps one fixed location; a positive value samples a new location after that
many optimization steps and carries the best patch to the new location.

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
$ source bash/run_PatchRS_realistic_non_targeted.sh  # for PatchRS
$ source bash/run_CamoPatch_realistic_non_targeted.sh  # for CamoPatch
```

#### Directed (_targeted_) attacks under the _realistic_ settings (i.e., attacking on 1000 _raw_ images)
```shell
$ source bash/run_DiVA_Patch_realistic_targeted.sh  # for DiVA-Patch (ours)
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
