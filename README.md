# SSC-RS

## SSC-RS: Elevate LiDAR Semantic Scene Completion with Representation Separation and BEV Fusion

This repository is for SSC-RS introduced in the following paper. [[arxiv paper]](https://arxiv.org/abs/2306.15349)

<!-- If you find our work useful, please cite

```

```
-->
## Preperation

### Prerequisites
Tested with
* python 3.7.10
* numpy 1.20.2
* torch 1.6.0
* torchvision 0.7.0
* torch-scatter 2.0.8
* spconv 1.1

### Dataset

Please download the Semantic Scene Completion dataset (v1.1) from the [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html) and extract it.

Or you can use [voxelizer](https://github.com/jbehley/voxelizer) to generate ground truths of semantic scene completion.

The dataset folder should be organized as follows.
```angular2
SemanticKITTI
├── dataset
│   ├── sequences
│   │  ├── 00
│   │  │  ├── labels
│   │  │  ├── velodyne
│   │  │  ├── voxels
│   │  │  ├── [OTHER FILES OR FOLDERS]
│   │  ├── 01
│   │  ├── ... ...
```

## Getting Start
Clone the repository:
```
git clone https://github.com/Jieqianyu/SSC-RS.git
```

We provide training routine examples in the `cfgs` folder. Make sure to change the dataset path to your extracted dataset location in such files if you want to use them for training. Additionally, you can change the folder where the performance and states will be stored.
* `config_dict['DATASET']['DATA_ROOT']` should be changed to the root directory of the SemanticKITTI dataset (`/.../SemanticKITTI/dataset/sequences`)
* `config_dict['OUTPUT']['OUT_ROOT'] ` should be changed to desired output folder.

### Train SSC-RS Net

```
$ cd <root dir of this repo>
$ python train.py --cfg cfgs/DSC-Base.yaml --dset_root <path/dataset/root>
```
### Validation

Validation passes are done during training routine. Additional pass in the validation set with saved model can be done by using the `validate.py` file. You need to provide the path to the saved model and the dataset root directory.

```
$ cd <root dir of this repo>
$ python validate.py --weights </path/to/model.pth> --dset_root <path/dataset/root>
```
### Test

Since SemantiKITTI contains a hidden test set, we provide test routine to save predicted output in same format of SemantiKITTI, which can be compressed and uploaded to the [SemanticKITTI Semantic Scene Completion Benchmark](http://www.semantic-kitti.org/tasks.html#ssc).

We recommend to pass compressed data through official checking script provided in the [SemanticKITTI Development Kit](http://www.semantic-kitti.org/resources.html#devkit) to avoid any issue.

You can provide which checkpoints you want to use for testing. We used the ones that performed best on the validation set during training. For testing, you can use the following command.

```
$ cd <root dir of this repo>
$ python test.py --weights </path/to/model.pth> --dset_root <path/dataset/root> --out_path <predictions/output/path>
```
### Pretrained Model

You can download the models with the scores below from this [Google drive link](https://drive.google.com/file/d/1-b3O7QS6hBQIGFTO-7qSG7Zb9kbQuxdO/view?usp=sharing),

| Model  | Segmentation | Completion |
|--|--|--|
| SSC-RS | 24.2 | 59.7 |

<sup>*</sup> Results reported to SemanticKITTI: Semantic Scene Completion leaderboard ([link](https://codalab.lisn.upsaclay.fr/competitions/7170\#results)).

## Acknowledgement
This project is not possible without multiple great opensourced codebases.
* [spconv](https://github.com/traveller59/spconv)
* [LMSCNet](https://github.com/cv-rits/LMSCNet)
* [SSA-SC](https://github.com/jokester-zzz/SSA-SC)
* [GASN](https://github.com/ItIsFriday/PcdSeg)
