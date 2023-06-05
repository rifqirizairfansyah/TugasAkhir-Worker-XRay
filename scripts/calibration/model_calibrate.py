#!/usr/bin/env python
# coding: utf-8

import os, sys
sys.path.insert(0, "..")
sys.path.insert(0, "/workspace/update/torchxrayvision")
sys.path.insert(0, "/workspace/update/torchxrayvision/torchxrayvision")
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torchvision, torchvision.transforms
import skimage.transform

import pickle
import random
import train_utils

import torchxrayvision as xrv

from os.path import basename
from jsonmerge import merge

from scripts.calibration.model_config import model_config


def main():

    config = config_data(transform_config="cropped", model_config=model_config)
    model_name = [
        "all_cropped", "all_cropped_v1", "all_cropped_relabelled-nih",
        "all_cropped_relabelled-nih_v1"
    ]
    weight_name = [
        basename(config["all"]["weights_url"]),
        basename(config["all_cropped"]["weights_url"]),
        basename(config["all_cropped_v1"]["weights_url"]),
        basename(config["all_cropped_relabelled-nih"]["weights_url"]),
        basename(config["all_cropped_relabelled-nih_v1"]["weights_url"]),
        basename(config["first"]["weights_url"]),
        basename(config["relabelled-nih"]["weights_url"]),
        basename(config["mixed-normal-contour"]["weights_url"])
    ]

    dataset_name = [
        "chex-nih-mimic_ch-pc-google-openi-kaggle",
        "chex-nih-mimic_ch-pc-google-openi-kaggle-mimic_nb",
        "chex-nih_relabel-mimic_ch-pc-google-openi-kaggle",
        "chex-nih_relabel-mimic_ch-pc-google-openi-kaggle-mimic_nb"
    ]

    for count, model in enumerate(model_name):

        for dataset in dataset_name:

            #weight = weight_name[count]
            weight = basename(config[model]["weights_url"])
            weight_base = join("/raid/COVID19/models/models_rad_findings",
                               config[model]["base"], "")

            process_calibration(model, weight_base, weight, dataset, config)


def config_data(transform_config, model_config):

    data_aug = None

    if transform_config == "standard":
        transforms = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(),
             xrv.datasets.XRayResizer(224)])

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

    elif transform_config == "cropped":
        transforms = torchvision.transforms.Compose(
            [xrv.datasets.XRayResizer(224)])

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

    dataset_type = {"type": transform_config}

    concate_config = merge(model_config, dataset_type)
    concate_config_1 = merge(concate_config, dataset_config)

    return concate_config_1


def process_calibration(model, weight_base, weight_name, dataset_name, config):

    weight_path = weight_base + weight_name

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default="", help='')
    parser.add_argument('--dataset', type=str, default=dataset_name)
    parser.add_argument('--weights_filename', type=str, default=weight_path)
    parser.add_argument('-seed', type=int, default=0, help='')
    parser.add_argument('-cuda', type=bool, default=True, help='')
    parser.add_argument('-batch_size', type=int, default=256, help='')
    parser.add_argument('-threads', type=int, default=12, help='')

    cfg = parser.parse_args()

    filename = "results_" + os.path.basename(
        cfg.weights_filename).split(".")[0] + "_" + dataset_name + ".pkl"

    datas = []
    datas_names = []

    if "nih_relabel" in cfg.dataset:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=config["nih_relabel"]["imgpath"],
            csvpath=config["nih_relabel"]["csvpath"],
            transform=config["nih_relabel"]["transform"],
            data_aug=config["nih_relabel"]["data_aug"])
        datas.append(dataset)
        datas_names.append("nih_relabel")

    if "nih" in cfg.dataset:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=config["nih"]["imgpath"],
            transform=config["nih"]["transform"],
            data_aug=config["nih"]["data_aug"])
        datas.append(dataset)
        datas_names.append("nih")

    if "pc" in cfg.dataset:
        dataset = xrv.datasets.PC_Dataset(imgpath=config["pc"]["imgpath"],
                                          transform=config["pc"]["transform"],
                                          data_aug=config["pc"]["data_aug"])
        datas.append(dataset)
        datas_names.append("pc")

    if "chex" in cfg.dataset:
        dataset = xrv.datasets.CheX_Dataset(
            imgpath=config["chex"]["imgpath"],
            csvpath=config["chex"]["csvpath"],
            transform=config["chex"]["transform"],
            data_aug=config["chex"]["data_aug"])
        datas.append(dataset)
        datas_names.append("chex")

    if "google" in cfg.dataset:
        dataset = xrv.datasets.NIH_Google_Dataset(
            imgpath=config["google"]["imgpath"],
            transform=config["google"]["transform"],
            data_aug=config["google"]["data_aug"])
        datas.append(dataset)
        datas_names.append("google")

    if "mimic_ch" in cfg.dataset:
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=config["mimic_ch"]["imgpath"],
            csvpath=config["mimic_ch"]["csvpath"],
            metacsvpath=config["mimic_ch"]["metacsvpath"],
            transform=config["mimic_ch"]["transform"],
            data_aug=config["mimic_ch"]["data_aug"])
        datas.append(dataset)
        datas_names.append("mimic_ch")

    if "mimic_nb" in cfg.dataset:
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=config["mimic_nb"]["imgpath"],
            csvpath=config["mimic_nb"]["csvpath"],
            metacsvpath=config["mimic_nb"]["metacsvpath"],
            transform=config["mimic_nb"]["transform"],
            data_aug=config["mimic_nb"]["data_aug"])
        datas.append(dataset)
        datas_names.append("mimic_nb")

    if "openi" in cfg.dataset:
        dataset = xrv.datasets.Openi_Dataset(
            imgpath=config["openi"]["imgpath"],
            transform=config["openi"]["transform"],
            data_aug=config["openi"]["data_aug"])
        datas.append(dataset)
        datas_names.append("openi")

    if "kaggle" in cfg.dataset:
        dataset = xrv.datasets.Kaggle_Dataset(
            imgpath=config["kaggle"]["imgpath"],
            transform=config["kaggle"]["transform"],
            data_aug=config["kaggle"]["data_aug"])
        datas.append(dataset)
        datas_names.append("kaggle")

    print("datas_names", datas_names)

    for d in datas:
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d)

    #cut out training sets
    train_datas = []
    test_datas = []
    for i, dataset in enumerate(datas):
        train_size = int(0.5 * len(dataset))
        test_size = len(dataset) - train_size
        torch.manual_seed(0)
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        #disable data aug
        test_dataset.data_aug = None

        #fix labels
        train_dataset.labels = dataset.labels[train_dataset.indices]
        test_dataset.labels = dataset.labels[test_dataset.indices]

        train_dataset.pathologies = dataset.pathologies
        test_dataset.pathologies = dataset.pathologies

        train_datas.append(train_dataset)
        test_datas.append(test_dataset)

    if len(datas) == 0:
        raise Exception("no dataset")
    elif len(datas) == 1:
        train_dataset = train_datas[0]
        test_dataset = test_datas[0]
    else:
        print("merge datasets")
        train_dataset = xrv.datasets.Merge_Dataset(train_datas)
        test_dataset = xrv.datasets.Merge_Dataset(test_datas)

    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("train_dataset.labels.shape", train_dataset.labels.shape)
    print("test_dataset.labels.shape", test_dataset.labels.shape)

    # load model
    model = torch.load(cfg.weights_filename, map_location='cpu')

    if cfg.cuda:
        model = model.cuda()

    print(model)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=False,
                                              num_workers=cfg.threads,
                                              pin_memory=cfg.cuda)

    results = train_utils.valid_test_epoch("test",
                                           0,
                                           model,
                                           "cuda",
                                           test_loader,
                                           torch.nn.BCEWithLogitsLoss(),
                                           limit=99999999)

    print(filename)

    pickle.dump(results, open(filename, "bw"))

    print("Done")


if __name__ == "__main__":
    main()