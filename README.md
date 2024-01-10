# S2M-TEA

This is the Pytorch implementation for the paper "Towards Assessing the Synthetic-to-Measured Adversarial Vulnerability of SAR ATR" (under review). It will be available soon.  <br>

## Environment

Pytorch 2.0.1, torchvision 0.15.2, kornia, skopt (scikit-optimize). <br>

## Resource

Please download [model weights](https://pan.baidu.com/s/1-Ktj6wdxcGZ5BdSRZDZQ1Q?pwd=5631) and the [SAMPLE synthetic-measured data pairs](https://pan.baidu.com/s/11RNkx0ArmktF-pVY829y0Q?pwd=5631) and put them into './models/' and './dataset/', respectively. <br>


## Synthetic-to-Measured Transfer (S2M) evaluation of surrogates and attack algorithms 

```
python transfer_eval.py --epsilon --surrogate_model --attack
```

## Perform a random Transferability Estimation Attack (TEA)

```
python TEA.py  --surrogate_model --sigma --lambda_ --samples
#this prints the architecture hyper-parameters and the fine-tuned weights will be saved at './FT_result/'

python transfer_eval.py --epsilon --local --surrogate_model --attack --beta --decay
```

## Details
The released code supports [PGD](https://arxiv.org/abs/1706.06083), [MI](https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html), [NI](https://arxiv.org/abs/1908.06281), [VT](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.html), [DI](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Improving_Transferability_of_Adversarial_Examples_With_Input_Diversity_CVPR_2019_paper.html), [TI](https://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Evading_Defenses_to_Transferable_Adversarial_Examples_by_Translation-Invariant_Attacks_CVPR_2019_paper.html), [SVA](https://ieeexplore.ieee.org/abstract/document/9800917), [Mixup-Attack](https://ieeexplore.ieee.org/abstract/document/9726211), and [ResNet18](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), ResNet34, [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html).

## Acknowledgement

This repository benefits a lot from the work listed below:

[Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) <br>
[ZhengyuZhao/TransferAttackEval](https://github.com/ZhengyuZhao/TransferAttackEval)<br>
[benjaminlewis-afrl/SAMPLE_dataset_public](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public)
[YonghaoXu/UAE-RS](https://github.com/YonghaoXu/UAE-RS)

