
# TayFCS
Implementation of "TayFCS: Towards Light Feature Combination Selection for Deep Recommender Systems," published at `KDD 2025` [[Paper](https://arxiv.org/abs/2507.03895)].

## Clone FuxiCTR and Use Our Code
Clone the [FuxiCTR](https://github.com/reczoo/FuxiCTR) repository to your local machine with the following command:
```
git clone https://github.com/reczoo/FuxiCTR.git
```

Then, copy the code from our repository into the `FuxiCTR/fuxictr/` and `FuxiCTR/model_zoo/` directories.

## Dataset Preparation
For Frappe and Avazu (We use the Avazu_x4_**002** version, referred to as avazu_x6 in our configuration files), you can directly download them from [FuxiCTR-Datasets](https://github.com/reczoo/Datasets/tree/main). Detailed download and configuration instructions are provided there.

For the iPinYou dataset, the original dataset processing treated the test set as the validation set, which we believe is not very reasonable. Therefore, we additionally split a portion of the training set as the test set. The specific code is located at `FuxiCTR/data/ipinyou_process.py`. 
Before using it, rename the downloaded iPinYou dataset's `train.csv` to `train_old.csv`, and then run the Python file.



For the above four datasets, please place the downloaded data in the `FuxiCTR/data` folder. You can refer to the corresponding dataset folder names in the `FuxiCTR/model_zoo/DNN/DNN_torch/config/dataset_config.yaml` file.


## TayScorer (Example based on DNN)
First, execute
```
cd model_zoo/DNN/DNN_torch
python run_expid_tayscorer.py --expid DNN_taylor_[frappe/ipinyou/avazu] --gpu 0
```
For example, if you want to run the experiment on the `Avazu` dataset, set the `expid` to `DNN_taylor_avazu` (the same applies below).
This will generate the feature combinations importance list.

The procedure for using Wide & Deep and DeepFM is exactly the same, so it will not be repeated here.


## LRE
Based on the obtained list, use Logistic regression for redundancy elimination.
```
cd model_zoo/LR
python run_redundancy_eliminator.py --expid LR_[frappe/ipinyou/avazu] --gpu 0
```


## Retraining with Feature Combinations
After placing the importance list and the LRE results in the specified folder, to retrain the base model (e.g., DNN), use:
```
cd model_zoo/DNN/DNN_torch
python run_expid_incre.py --expid DNN_re_[frappe/ipinyou/avazu] --gpu 0 --com_num [10]
```
For transfer learning (e.g., MaskNet), use:
```
cd model_zoo/MaskNet
python run_expid_incre.py --expid MaskNet_re_[frappe/ipinyou/avazu] --gpu 0 --com_num [10]
```


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
