## CLIP-based Fusion-modal Reconstructing Hashing for Unsupervised Large-scale Cross-modal Retrieval

This repository contains the author's implementation in PyTorch for the IJMIR-23 paper "CLIP-based Fusion-modal Reconstructing Hashing for Unsupervised Large-scale Cross-modal Retrieval".

## Introduction

Cross-modal hashing encodes the multimedia data into a common binary hash space in which the correlations among the samples from different modalities can be effectively measured. Deep cross-modal hashing further improves the retrieval performance as the deep neural networks can generate more semantically relevant features and hash codes. Currently, the existing unsupervised hashing methods generally have two limitations: (1) Existing methods fail to adequately capture the latent semantic relevance and coexistent information from the different modality data. (2) Existing unsupervised methods typically construct a similarity matrix to guide the hash code learning, which suffers from inaccurate similarity problems, resulting in sub-optimal retrieval performance. To address these issues, we propose a novel CLIP-based Fusion-modal Reconstructing Hashing (CFRH) for Large-scale Unsupervised Cross-modal Retrieval. First, we use CLIP to encode cross-modal features of visual modality and learn the common representation space of the hash code using modality-specific autoencoders. Second, we propose an efficient fusion approach to construct a semantically complementary affinity matrix that can maximize the potential semantic relevance of different modal instances. Furthermore, to retain the intrinsic semantic similarity of all similar pairs in the learned hash codes, an objective function for similarity reconstructing based on semantic complementation is designed to learn high-quality hash code representations. 

<div align=center><img src="https://github.com/AwakerLee/CFRH/blob/main/Fig31.jpg" width="90%" height="90%"></div align=center>

***********************************************************************************************************
## Dependencies

Please, install the following packages:

- Python (>=3.8)
- pytorch
- torchvision
- h5py
- CLIP

## Datasets
You can download the features of the datasets from:
For datasets, we follow [Deep Cross-Modal Hashing's Github (Jiang, CVPR 2017)](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_matlab/DCMH_matlab). You can download these datasets from:
- Wikipedia articles, [Link](http://www.svcl.ucsd.edu/projects/crossmodal/)
- MIRFLICKR25K, [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EpLD8yNN2lhIpBgQ7Kl8LKABzM68icvJJahchO7pYNPV1g?e=IYoeqn)], [[Baidu Pan](https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew), password: 8dub]
- NUS-WIDE (top-10 concept), [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EoPpgpDlPR1OqK-ywrrYiN0By6fdnBvY4YoyaBV5i5IvFQ?e=kja8Kj)], [[Baidu Pan](https://pan.baidu.com/s/1GFljcAtWDQFDVhgx6Jv_nQ), password: ml4y]
 - MS-COCO, [BaiduPan(password: 5uvp)](https://pan.baidu.com/s/1uoV4K1mBwX7N1TVmNEiPgA)
 
## Implementation

Here we provide the implementation of our proposed models, along with datasets. The repository is organised as follows:

 - `data/` contains the necessary dataset files for NUS-WIDE, MIRFlickr, and MS-COCO;
 - `models.py` contains the implementation of the model;
 
 Finally, `main.py` puts all of the above together and can be used to execute a full training run on MIRFlcikr or NUS-WIDE or MS-COCO.

## Process
 - Place the datasets in `data/`
 - Set the experiment parameters in `main.py`.
 - Train a model:
 ```bash
 python main.py
```
 - Modify the parameter `EVAL = True` in `main.py` for evaluation:
  ```bash
 python main.py
```

## Citation
If you find our work or the code useful, please consider cite our paper using:
```bash
@article{mingyong2023clip,
  title={CLIP-based fusion-modal reconstructing hashing for large-scale unsupervised cross-modal retrieval},
  author={Mingyong, Li and Yewen, Li and Mingyuan, Ge and Longfei, Ma},
  journal={International Journal of Multimedia Information Retrieval},
  volume={12},
  number={1},
  pages={2},
  year={2023},
  publisher={Springer}
}
```
