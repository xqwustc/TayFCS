
# TayFCS
Implementation of "TayFCS: Towards Light Feature Combination Selection for Deep Recommender Systems," published at `KDD 2025` [[Paper](https://arxiv.org/abs/2507.03895)].

## Clone FuxiCTR and Use Our Code
Clone the [FuxiCTR](https://github.com/reczoo/FuxiCTR) repository to your local machine with the following command:
```
git clone https://github.com/reczoo/FuxiCTR.git
```

Then, copy the code from our repository into the `FuxiCTR/model_zoo/` directory.

## Dataset Preparation
For Frappe and Avazu (We use the Avazu_x4_**002** version， 在我们的配置文件中称为avazu_x6), you can directly download them from [FuxiCTR-Datasets](https://github.com/reczoo/Datasets/tree/main). Detailed download and configuration instructions are provided there.

For the iPinYou dataset, the original dataset processing treated the test set as the validation set, which we believe is not very reasonable. Therefore, we additionally split a portion of the training set as the test set. The specific code is as follows:


For the above four datasets, please place the downloaded data in the `FuxiCTR/data` folder. You can refer to the corresponding dataset folder names in the `FuxiCTR/model_zoo/DNN/DNN_torch/config/dataset_config.yaml` file.

## Pretraining the Models


## TayScorer


## RFE


## Retraining with Feature Combinations

### Contact Us & Acknowledgments
If you have any questions, feel free to contact Xianquan Wang (email: wxqcn@mail.ustc.edu.cn).
Our work heavily relies on the excellent contributions of [FuxiCTR](https://github.com/reczoo/FuxiCTR). We sincerely thank the team for their efforts.

Cite us if our work helps you~
```bibtex
@inproceedings{wang2025tayfcs,
  title={TayFCS: Towards Light Feature Combination Selection for Deep Recommender Systems},
  author={Wang, Xianquan and Du, Zhaocheng and Zhu, Jieming and Wu, Chuhan and Jia, Qinglin and Dong, Zhenhua},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={5007--5017},
  year={2025}
}
```
