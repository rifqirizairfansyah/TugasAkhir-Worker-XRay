# **Validation of Radiological Findings Model** 

## Pathologies
```
default_pathologies = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
    'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture',
    'Lung Opacity', 'Enlarged Cardiomediastinum'
]
```

## Dataset
### Dataset configuration

**standard dataset :**

```
dataset_config = {
    "nih_relabel": {
        "imgpath": "/raid/COVID19/nih-dataset/images",
        "csvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/nih_train_relabeled_standard-format.csv",
        "transform": transforms,
        "data_aug": data_aug
    },
    "nih": {
        "imgpath": "/raid/COVID19/nih-dataset/images",
        "transform": transforms,
        "data_aug": data_aug
    },
    "pc": {
        "imgpath": "/raid/COVID19/padchest/padchest_resize/images-224",
        "transform": transforms,
        "data_aug": data_aug
    },
    "chex": {
        "imgpath": "/raid/COVID19/chexpert/",
        "csvpath": "/raid/COVID19/chexpert/CheXpert-v1.0/train.csv",
        "transform": transforms,
        "data_aug": data_aug
    },
    "google": {
        "imgpath": "/raid/COVID19/nih-dataset/images",
        "transform": transforms,
        "data_aug": data_aug
    },
    "mimic_ch": {
        "imgpath":
        "//raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
        "csvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-chexpert.csv.gz",
        "metacsvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
        "transform": transforms,
        "data_aug": data_aug
    },
    "mimic_nb": {
        "imgpath":
        "/raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
        "csvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-negbio.csv.gz",
        "metacsvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
        "transform": transforms,
        "data_aug": data_aug
    },
    "openi": {
        "imgpath": "/raid/COVID19/openi/png",
        "transform": transforms,
        "data_aug": data_aug
    },
    "kaggle": {
        "imgpath":
        "/raid/COVID19/rsna_pneumonia/JPG/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
        "transform": transforms,
        "data_aug": data_aug
    }
}
```

**cropped dataset :**

```
dataset_config = {
    "nih_relabel": {
        "imgpath": "/raid/COVID19/nih-dataset/images_cropped",
        "csvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/nih_train_relabeled_standard-format.csv",
        "transform": transforms,
        "data_aug": data_aug
    },
    "nih": {
        "imgpath": "/raid/COVID19/nih-dataset/images_cropped",
        "transform": transforms,
        "data_aug": data_aug
    },
    "pc": {
        "imgpath":
        "/raid/COVID19/padchest/padchest_resize/images-224-cropped",
        "transform": transforms,
        "data_aug": data_aug
    },
    "chex": {
        "imgpath": "/raid/COVID19/chexpert/",
        "csvpath":
        "/raid/COVID19/chexpert/CheXpert-v1.0/train_cropped_concate.csv",
        "transform": transforms,
        "data_aug": data_aug
    },
    "google": {
        "imgpath": "/raid/COVID19/nih-dataset/images_cropped",
        "transform": transforms,
        "data_aug": data_aug
    },
    "mimic_ch": {
        "imgpath":
        "/raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files-cropped",
        "csvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-chexpert.csv.gz",
        "metacsvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
        "transform": transforms,
        "data_aug": data_aug
    },
    "mimic_nb": {
        "imgpath":
        "/raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files-cropped",
        "csvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-negbio.csv.gz",
        "metacsvpath":
        "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
        "transform": transforms,
        "data_aug": data_aug
    },
    "openi": {
        "imgpath": "/raid/COVID19/openi/png-cropped",
        "transform": transforms,
        "data_aug": data_aug
    },
    "kaggle": {
        "imgpath":
        "/raid/COVID19/rsna_pneumonia/JPG/kaggle-pneumonia-jpg/stage_2_train_images_cropped",
        "transform": transforms,
        "data_aug": data_aug
    }
}
```

### Dataset Function
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
## Run validation script

```
python3 etoe_valid_model.py
```
this script will produce pkl files for each configuration of model and validation dataset.
## Process pkl files using jupyter notebook
forwarding port in 6664
```
jupyter notebook --allow-root --no-browser --port 6664 --NotebookApp.token='' --ip 0.0.0.0 
```
this notebook processes pkl files and extract information from pkl file. The information is constructed in dataframe and saved in csv formated file