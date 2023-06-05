# torchxrayvision

A library for chest X-ray datasets and models. Including pre-trainined models.

This code is still under development

## Getting started

```
pip install torchxrayvision

import torchxrayvision as xrv
```

These are default pathologies:
```
xrv.datasets.default_pathologies 

['Atelectasis',
 'Consolidation',
 'Infiltration',
 'Pneumothorax',
 'Edema',
 'Emphysema',
 'Fibrosis',
 'Effusion',
 'Pneumonia',
 'Pleural_Thickening',
 'Cardiomegaly',
 'Nodule',
 'Mass',
 'Hernia',
 'Lung Lesion',
 'Fracture',
 'Lung Opacity',
 'Enlarged Cardiomediastinum']
```

## Call Models

Specify weights for pretrained models (currently all DenseNet121)
Note: Each pretrained model has 18 outputs. The `all` model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset. The only valid outputs are listed in the field `{dataset}.pathologies` on the dataset that corresponds to the weights. 

```
model = xrv.models.DenseNet(weights="all")
model = xrv.models.DenseNet(weights="kaggle")
model = xrv.models.DenseNet(weights="nih")
model = xrv.models.DenseNet(weights="chex")
model = xrv.models.DenseNet(weights="minix_nb")
model = xrv.models.DenseNet(weights="minix_ch")

```

## Run development environment using docker

```
nvidia-docker run -ti --env LICENSE=yes \
--rm -it --name pneumonia_severity_eval \
-p 6664:6664 -p 5002:5002 -p 5003:5003 -p 5004:6664  \
-v /home/yudi/workspace/covid/update/torchxrayvision/:/workspace/update/torchxrayvision/ \
-v /home/yudi/workspace/covid/model_server/pneumonia_severity_serving/:/workspace/pneumonia_severity_serving/ \
-v /raid/COVID19/:/raid/COVID19/ \
-v /mnt/nfs-covid/:/mnt/nfs-covid/ \
-v /raid/COVID19/models/models_rad_findings/train-02-08-2020/:/root/.torchxrayvision/models_data/train-02-08-2020/ \
-v /raid/COVID19/models/models_rad_findings/train-cropped/:/root/.torchxrayvision/models_data/train-cropped/ \
-v /raid/COVID19/models/models_rad_findings/train-cropped-relabelled-nih/:/root/.torchxrayvision/models_data/train-cropped-relabelled-nih/ \
-v /raid/COVID19/models/models_rad_findings/train-lung-contour/:/root/.torchxrayvision/models_data/train-lung-contour/ \
-v /raid/COVID19/models/models_rad_findings/train-mixed-lung-contour/:/root/.torchxrayvision/models_data/train-mixed-lung-contour/ \
-v /raid/COVID19/models/models_rad_findings/train-relabelled-nih/:/root/.torchxrayvision/models_data/train-relabelled-nih/ \
-v /home/yudi/workspace/covid/evaluation/etoe_evaluation/:/workspace/etoe_evaluation/ \
-e NVIDIA_VISIBLE_DEVICES=3,4 \
-w /workspace/ \
risetai/covid:pneumonia-severity-serving /bin/bash
```

```
jupyter notebook --allow-root --no-browser --port 6663 --NotebookApp.token='' --ip 0.0.0.0 
```

## Datasets
Only stats for PA/AP views are shown. Datasets may include more.

```
transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])

d_kaggle = xrv.datasets.Kaggle_Dataset(imgpath="path to stage_2_train_images_jpg",
                                       transform=transform)
                
d_chex = xrv.datasets.CheX_Dataset(imgpath="path to CheXpert-v1.0-small",
                                   csvpath="path to CheXpert-v1.0-small/train.csv",
                                   transform=transform)

d_nih = xrv.datasets.NIH_Dataset(imgpath="path to NIH images")

d_nih2 = xrv.datasets.NIH_Google_Dataset(imgpath="path to NIH images")

d_pc = xrv.datasets.PC_Dataset(imgpath="path to image folder")


d_covid19 = xrv.datasets.COVID19_Dataset() # specify imgpath and csvpath for the dataset
```

National Library of Medicine Tuberculosis Datasets [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)

```
d_nlmtb = xrv.datasets.NLMTB_Dataset(imgpath="path to MontgomerySet or ChinaSet_AllFiles")
```

list of datasets path :

1. RSNA Pneumonia : /raid/COVID19/rsna_pneumonia
2. Chexpert : /raid/COVID19/chexpert/CheXpert-v1.0
3. NIH : /raid/COVID19/nih-dataset
4. NIH-Google : /raid/COVID19/nih-google-dataset
5. PadChest : /raid/COVID19/padchest/padchest_ori/PADCHEST_SJ/image_zips/images
6. OpenI : /raid/COVID19/openi/png
7. MIMIC : /raid/COVID19/mimic


## Model Configuration
To call model configuration, 
```
from torchxrayvision.model import model_urls as model_config
```
model_urls contains kind of configuration below, 
```
model_urls['all'] = {
    "weights_url":
    'https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "op_threshs": [
        0.07422872, 0.038290843, 0.09814756, 0.0098118475, 0.023601074,
        0.0022490358, 0.010060724, 0.103246614, 0.056810737, 0.026791653,
        0.050318155, 0.023985857, 0.01939503, 0.042889766, 0.053369623,
        0.035975814, 0.20204692, 0.05015312
    ],
    "ppv80_thres": [
        0.72715247, 0.8885005, 0.92493945, 0.6527224, 0.68707734, 0.46127197,
        0.7272054, 0.6127343, 0.9878492, 0.61979693, 0.66309816, 0.7853459,
        0.930661, 0.93645346, 0.6788558, 0.6547198, 0.61614525, 0.8489876
    ],
    "base":
    "cohen",
    "area_opacity": {
        "theta": 0.9342979788780212,
        "bias": 3.696541
    },
    "degree_opacity": {
        "theta": 0.5484423041343689,
        "bias": 2.5535977
    }
}

model_urls['nih'] = {
    "weights_url":
    'https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "op_threshs": [
        0.039117552, 0.0034529066, 0.11396341, 0.0057298196, 0.00045666535,
        0.0018880932, 0.012037827, 0.038744126, 0.0037213727, 0.014730946,
        0.016149804, 0.054241467, 0.037198864, 0.0004403434, np.nan, np.nan,
        np.nan, np.nan
    ],
    "base":
    "cohen"
}

model_urls['pc'] = {
    "weights_url":
    'https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "op_threshs": [
        0.031012505, 0.013347598, 0.081435576, 0.001262615, 0.002587246,
        0.0035944257, 0.0023071, 0.055412333, 0.044385884, 0.042766232,
        0.043258056, 0.037629247, 0.005658899, 0.0091741895, np.nan,
        0.026507627, np.nan, np.nan
    ],
    "base":
    "cohen"
}

model_urls['chex'] = {
    "weights_url":
    'https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "op_threshs": [
        0.1988969, 0.05710573, np.nan, 0.0531293, 0.1435217, np.nan, np.nan,
        0.27212676, 0.07749717, np.nan, 0.19712369, np.nan, np.nan, np.nan,
        0.09932402, 0.09273402, 0.3270967, 0.10888247
    ],
    "base":
    "cohen"
}

model_urls['kaggle'] = {
    "weights_url":
    'https://github.com/mlmed/torchxrayvision/releases/download/v1/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "op_threshs": [
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        0.13486601, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        0.13511065, np.nan
    ],
    "base":
    "cohen"
}

model_urls['mimic_nb'] = {
    "weights_url":
    'https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "op_threshs": [
        0.08558747, 0.011884617, np.nan, 0.0040595434, 0.010733786, np.nan,
        np.nan, 0.118761964, 0.022924708, np.nan, 0.06358637, np.nan, np.nan,
        np.nan, 0.022143636, 0.017476924, 0.1258702, 0.014020768
    ],
    "base":
    "cohen"
}

model_urls['mimic_ch'] = {
    "weights_url":
    'https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "op_threshs": [
        0.09121389, 0.010573786, np.nan, 0.005023008, 0.003698257, np.nan,
        np.nan, 0.08001232, 0.037242252, np.nan, 0.05006329, np.nan, np.nan,
        np.nan, 0.019866971, 0.03823637, 0.11303808, 0.0069147074
    ],
    "base":
    "cohen"
}

model_urls['all_cropped'] = {
    "weights_url":
    'chex-nih-mimic_nb-mimic_ch-pc-google-openi-kaggle-densenet121-model_2_cropped-best.pt',
    "op_threshs": [
        0.08653892, 0.017242203, 0.061408978, 0.006965817, 0.007230344,
        0.008352524, 0.0052154628, 0.09193552, 0.035084374, 0.020144384,
        0.0809918, 0.032811552, 0.0071497215, 0.014541325, 0.019512547,
        0.031239916, 0.11984462, 0.00936549
    ],
    "ppv80_thres": [
        0.9937518, 0.99484503, 0.57375365, 0.9282644, 0.9529527, 0.5661315,
        0.4997312, 0.91090864, 0.99205655, 0.8359465, 0.9208461, 0.91701555,
        0.9997881, 0.8658175, 0.99993837, 0.99887985, 0.8762074, 0.9988675
    ],
    "base":
    "train-cropped"
}

model_urls['all_cropped_v1'] = {
    "weights_url":
    'chex-nih-mimic_ch-pc-google-openi-kaggle-densenet121-cropped-best.pt',
    "op_threshs": [
        0.04718963, 0.015039951, 0.07342019, 0.0051146024, 0.007409899,
        0.008317836, 0.004983357, 0.04215132, 0.030943004, 0.031830408,
        0.039926834, 0.031946592, 0.0096772835, 0.016858106, 0.055931136,
        0.046968885, 0.15709805, 0.015543708
    ],
    "ppv80_thres": [
        0.8200604, 0.8610378, 0.64609003, 0.7353858, 0.96263915, 0.6738628,
        0.8235256, 0.5164257, 0.9624524, 0.85824424, 0.7721861, 0.91494757,
        0.9859875, 0.4577153, 0.9642816, 0.9072195, 0.6847318, 0.9756423
    ],
    "base":
    "train-cropped"
}

model_urls['all_cropped_relabelled-nih'] = {
    "weights_url":
    'chex-nih-mimic_nb-mimic_ch-pc-google-openi-kaggle-densenet121-model_2_cropped_relabel-nih-best.pt',
    "op_threshs": [
        0.043316294, 0.01043874, 0.07821479, 0.0030074122, 0.0113249775,
        0.010188991, 0.0059004985, 0.05317818, 0.037837196, 0.0050458475,
        0.09594208, 0.040771976, 0.007820424, 0.015169545, 0.006620609,
        0.029966863, 0.08549373, 0.006000353
    ],
    "ppv80_thres": [
        0.999503, 0.99491274, 0.9934208, 0.96971613, 0.9995702, 0.8879771,
        0.9983138, 0.890265, 0.9672664, 0.9856338, 0.98278266, 0.99870896,
        0.9999771, 0.9739644, 0.9996896, 0.9940188, 0.9450582, 0.9978744
    ],
    "base":
    "train-cropped-relabelled-nih"
}

model_urls['all_cropped_relabelled-nih_v1'] = {
    "weights_url":
    'chex-nih-mimic_ch-pc-google-openi-kaggle-densenet121-cropped_relabel-nih-best.pt',
    "op_threshs": [
        0.06420791, 0.011237458, 0.046035413, 0.0052339705, 0.011177873,
        0.0024271521, 0.0063031274, 0.06383512, 0.0338762, 0.02408131,
        0.06494272, 0.037538163, 0.0054588974, 0.026531212, 0.044195093,
        0.0327385, 0.17920227, 0.024809295
    ],
    "ppv80_thres": [
        0.93297035, 0.9559677, 0.80296314, 0.8045187, 0.94467974, 0.481314,
        0.9366123, 0.7138835, 0.96114165, 0.7748556, 0.8895693, 0.74475545,
        0.9937219, 0.9175519, 0.9321702, 0.84740615, 0.65827876, 0.93292886
    ],
    "base":
    "train-cropped-relabelled-nih",
    "area_opacity": {
        "theta": 1.0184412002563477,
        "bias": 5.3219767
    },
    "degree_opacity": {
        "theta": 0.5484423041343689,
        "bias": 2.5535977
    }
}

model_urls['first'] = {
    "weights_url":
    'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-first-best.pt',
    "labels": [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
        'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'
    ],
    "op_threshs": [
        0.05629035, 0.012764132, 0.0731739, 0.010680856, 0.01125753,
        0.0047965255, 0.0058362824, 0.06742897, 0.03591366, 0.015977284,
        0.07871004, 0.024436774, 0.0072485073, 0.037256896, 0.060763117,
        0.030289847, 0.24607852, 0.05840985
    ],
    "base":
    "train-02-08-2020"
}

model_urls['relabelled-nih'] = {
    "weights_url":
    'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-relabelled-nih-best.pt',
    "labels": [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
        'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'
    ],
    "op_threshs": [
        0.04834002, 0.013684599, 0.0818266, 0.0073450753, 0.012969672,
        0.0025104291, 0.014892542, 0.06347001, 0.03242855, 0.017052772,
        0.04459352, 0.05550588, 0.019552503, 0.040999617, 0.027115278,
        0.029708073, 0.15177177, 0.017925948
    ],
    "base":
    "train-relabelled-nih"
}

model_urls['mixed-normal-contour'] = {
    "weights_url":
    'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-mixed-normal-contour-best.pt',
    "labels": [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
        'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'
    ],
    "op_threshs": [
        0.092855476, 0.018703017, 0.067748964, 0.009171672, 0.0117160985,
        0.007753978, 0.0061168266, 0.07964283, 0.045002565, 0.03382119,
        0.08865285, 0.050985422, 0.018902548, 0.024114724, 0.040174063,
        0.044628836, 0.25872877, 0.031567562
    ],
    "base":
    "train-mixed-lung-contour"
}
```

## Dataset configuration 

### JSON based configuration for standard and cropped dataset
**standard :**
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
**cropped :**
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
### Inside called function for each dataset configuration

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
Mimic_Chexpert
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
Mimic_Negbio
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

## Dataset tools

relabel_dataset will align labels to have the same order as the pathologies argument.
```
xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies , d_nih) # has side effects
```

## Citation

```
Joseph Paul Cohen, Joseph Viviano, Mohammad Hashir, and Hadrien Bertrand. 
TorchXrayVision: A library of chest X-ray datasets and models. 
https://github.com/mlmed/torchxrayvision, 2020
```
and
```
Cohen, J. P., Hashir, M., Brooks, R., & Bertrand, H. 
On the limits of cross-domain generalization in automated X-ray prediction. 
Medical Imaging with Deep Learning 2020 (Online: [https://arxiv.org/abs/2002.02497](https://arxiv.org/abs/2002.02497))

@inproceedings{cohen2020limits,
  title={On the limits of cross-domain generalization in automated X-ray prediction},
  author={Cohen, Joseph Paul and Hashir, Mohammad and Brooks, Rupert and Bertrand, Hadrien},
  booktitle={Medical Imaging with Deep Learning}
  year={2020}
}
```
