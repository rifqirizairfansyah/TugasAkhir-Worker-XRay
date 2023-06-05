# **Training of Radiological Findings Model** 

## Pathologies
```
default_pathologies = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
    'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture',
    'Lung Opacity', 'Enlarged Cardiomediastinum'
]
```

## Dataset Configuration

```
parser.add_argument('--dataset', type=str,default="chex-nih-mimic_ch-pc-google-openi-kaggle")
```
Here is defined function for each dataset,

NIH Relabelled
```
if "nih" in cfg.dataset:
    dataset = xrv.datasets.NIH_Dataset(
        imgpath="/raid/COVID19/nih-dataset/images_cropped", 
        csvpath= "/workspace/update/torchxrayvision/torchxrayvision/nih_train_relabeled_standard-format.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("nih")
```
NIH standard 
```
if "nih" in cfg.dataset:
    dataset = xrv.datasets.NIH_Dataset(
        imgpath="/raid/COVID19/nih-dataset/images_cropped", 
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("nih")
```
Padchest
```
if "pc" in cfg.dataset:
    dataset = xrv.datasets.PC_Dataset(
        imgpath="/raid/COVID19/padchest/padchest_resize/images-224-cropped",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("pc")
```
Chexpert
```
if "chex" in cfg.dataset:
    dataset = xrv.datasets.CheX_Dataset(
        imgpath="/raid/COVID19/chexpert/",
        csvpath=
        "/raid/COVID19/chexpert/CheXpert-v1.0/train_cropped_concate.csv",
        transform=transforms,
        data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("chex")
```
NIH Google
```
if "google" in cfg.dataset:
    dataset = xrv.datasets.NIH_Google_Dataset(
        imgpath="/raid/COVID19/nih-dataset/images_cropped",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("google")
```
MIMIC_Chexpert
```
if "mimic_ch" in cfg.dataset:
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath=
        "/raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files-cropped",
        csvpath=
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-chexpert.csv.gz",
        metacsvpath=
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
        transform=transforms,
        data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("mimic_ch")
```
MIMIC_Negbio
```
if "mimic_nb" in cfg.dataset:
    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath=
        "/raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files-cropped",
        csvpath=
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-negbio.csv.gz",
        metacsvpath=
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
        transform=transforms,
        data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("mimic_nb")
```
OpenI
```
if "openi" in cfg.dataset:
    dataset = xrv.datasets.Openi_Dataset(
        imgpath="/raid/COVID19/openi/png-cropped",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("openi")
```
Kaggle RSNA Pneumonia
```
if "kaggle" in cfg.dataset:
    dataset = xrv.datasets.Kaggle_Dataset(
        imgpath="/raid/COVID19/rsna_pneumonia/JPG/kaggle-pneumonia-jpg/stage_2_train_images_cropped",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("kaggle")
```
## Model Configuration
```
parser.add_argument('--model', type=str, default="densenet121")
```
Here, they provide three backbone models for traning. Those are,
1. densenet : densenet-121, densenet-169, densenet-201, densenet-161
2. resnet101
3. shufflenet_v2_x2_0

## Hyperparameter configuration

```
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--num_epochs', type=int, default=160, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--threads', type=int, default=8, help='')
parser.add_argument('--taskweights', type=bool, default=True, help='')
parser.add_argument('--featurereg', type=bool, default=False, help='')
parser.add_argument('--weightreg', type=bool, default=False, help='')
parser.add_argument('--data_aug', type=bool, default=True, help='')
parser.add_argument('--label_concat', type=bool, default=False, help='')
parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
```

## Run training script
```
python3 train_model.py
```
this script is for training model from scratch using certain dataset combination and hyperparameter setting. The .pt model produced every step of training will be stored in specific location, cfg.output_dir.

