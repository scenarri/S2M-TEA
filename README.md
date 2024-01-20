# S2M-TEA

This is the Pytorch implementation for the paper "Towards Assessing the Synthetic-to-Measured Adversarial Vulnerability of SAR ATR" (under review).  <br>

## Environment

Pytorch 2.0.1, torchvision 0.15.2, kornia, skopt (scikit-optimize). <br>

## Preparation

Please download [model weights](https://pan.baidu.com/s/1-Ktj6wdxcGZ5BdSRZDZQ1Q?pwd=5631) and the [SAMPLE synthetic-measured data pairs](https://pan.baidu.com/s/11RNkx0ArmktF-pVY829y0Q?pwd=5631). They are meant to be located at './models/' and './dataset/', respectively. <br>


## Synthetic-to-Measured Transfer (S2M) evaluation
The released code supports [PGD](https://arxiv.org/abs/1706.06083), [MI](https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html), [NI](https://arxiv.org/abs/1908.06281), [VT](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.html), [DI](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Improving_Transferability_of_Adversarial_Examples_With_Input_Diversity_CVPR_2019_paper.html), [TI](https://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Evading_Defenses_to_Transferable_Adversarial_Examples_by_Translation-Invariant_Attacks_CVPR_2019_paper.html), [SVA](https://ieeexplore.ieee.org/abstract/document/9800917), [Mixup-Attack](https://ieeexplore.ieee.org/abstract/document/9726211), and [ResNet18](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), ResNet34, [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html).
The results will differ slightly from those reported in the paper due to seed setting. The command below generates adversarial perturbations based on the PGD attack and ResNet18 (trained over the synthetic data), and it reports the attack success rates (ASRs) against the measured data-trained victim models.
```
python transfer_eval.py --surrogate RN18/RN34/CNX --attack PGD --batch_size 128 
```

To reproduce the results in our paper, please use our finetuned models and trigger --AS using architecture hyper-parameters in the paper:
```
python transfer_eval.py --surrogate RN18_FT --AS
```
or input customized paras. with --beta and --decay if interested.

You can also load a surrogate by trigger --local and appoint the file name (in './FT_result/') with --surrogate, like
```
python transfer_eval.py --surrogate RN18_seed_12345.pth --local
```


## Perform a local Transferability Estimation Attack (TEA)

This prints the architecture hyper-parameters and saves the finetuned weights at './FT_result/', and these two surrogates (FT and FT+AS) will be evaluated with --test being triggered:

```
python TEA.py  --surrogate RN18 --sigmaFT 0.2 --sigmaAS 0.3 --lambda_ 1. --samples 5 --test
```
Here samples is the number of Gaussian neighbors being sampled during loss calculation.  A larger one leads to better stability and a smaller one favors lower memory usage.

Run architecture selection for a given surrogate (in './FT_result/'):

```
python TEA.py --name RN18_seed_12345.pth --test
```


## Acknowledgements

This repository benefits a lot from the works listed below:

[Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) <br>
[ZhengyuZhao/TransferAttackEval](https://github.com/ZhengyuZhao/TransferAttackEval)<br>
[benjaminlewis-afrl/SAMPLE_dataset_public](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public)<br>
[YonghaoXu/UAE-RS](https://github.com/YonghaoXu/UAE-RS)

