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
from torchxrayvision.models import model_urls as model_config

from os.path import basename
from jsonmerge import merge


def config_data(model_config=model_config, parser_config=None):

    valid_config = {
        "shuffle": parser_config.shuffle,
        "seed": parser_config.seed,
        "cuda": parser_config.cuda,
        "batch_size": parser_config.batch_size,
        "threads": parser_config.threads,
        "output_dir": parser_config.output_dir
    }

    data_aug = None
    views = parser_config.data_views
    transform_config = parser_config.data_type

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
                "data_aug": data_aug,
                "views": views
            },
            "nih": {
                "imgpath": "/raid/COVID19/nih-dataset/images",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "pc": {
                "imgpath": "/raid/COVID19/padchest/padchest_resize/images-224",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "chex": {
                "imgpath": "/raid/COVID19/chexpert/",
                "csvpath": "/raid/COVID19/chexpert/CheXpert-v1.0/train.csv",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "google": {
                "imgpath": "/raid/COVID19/nih-dataset/images",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "mimic_ch": {
                "imgpath":
                "//raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
                "csvpath":
                "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-chexpert.csv.gz",
                "metacsvpath":
                "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "mimic_nb": {
                "imgpath":
                "/raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
                "csvpath":
                "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-negbio.csv.gz",
                "metacsvpath":
                "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
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
                "data_aug": data_aug,
                "views": views
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
                "data_aug": data_aug,
                "views": views
            },
            "nih": {
                "imgpath": "/raid/COVID19/nih-dataset/images_cropped",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "pc": {
                "imgpath":
                "/raid/COVID19/padchest/padchest_resize/images-224-cropped",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "chex": {
                "imgpath": "/raid/COVID19/chexpert/",
                "csvpath":
                "/raid/COVID19/chexpert/CheXpert-v1.0/train_cropped_concate.csv",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "google": {
                "imgpath": "/raid/COVID19/nih-dataset/images_cropped",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "mimic_ch": {
                "imgpath":
                "/raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files-cropped",
                "csvpath":
                "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-chexpert.csv.gz",
                "metacsvpath":
                "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
            },
            "mimic_nb": {
                "imgpath":
                "/raid/COVID19/mimic/jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files-cropped",
                "csvpath":
                "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-negbio.csv.gz",
                "metacsvpath":
                "/workspace/update/torchxrayvision/torchxrayvision/mimic-cxr-2.0.0-metadata.csv.gz",
                "transform": transforms,
                "data_aug": data_aug,
                "views": views
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
                "data_aug": data_aug,
                "views": views
            }
        }

    dataset_type = {"type": transform_config}
    dataset_views = {"views": views}

    concate_config = merge(model_config, dataset_type)
    concate_config_1 = merge(concate_config, dataset_config)
    concate_config_2 = merge(concate_config_1, dataset_views)
    concate_config_3 = merge(concate_config_2, valid_config)

    return concate_config_3


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default="", help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cuda', type=bool, default=True, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('-threads', type=int, default=12, help='')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=
        "/workspace/update/torchxrayvision/scripts/validation/pkl_files/standard_dataset/v3"
    )
    parser.add_argument('--data_type', type=str, default="standard")
    parser.add_argument('--data_views', type=str, default="AP")

    return parser


def process_validation(model_name, weight, dataset_name, config):

    weights_filename = join(weight["base"], weight["name"])

    pkl_file = join(config["output_dir"],
                    ("valid_" + config["type"] + "_" + model_name + "_with_" +
                     dataset_name + ".pkl"))
    print(weights_filename, pkl_file)
    datas = []
    datas_names = []

    if "nih_relabel" in dataset_name:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=config["nih_relabel"]["imgpath"],
            csvpath=config["nih_relabel"]["csvpath"],
            transform=config["nih_relabel"]["transform"],
            data_aug=config["nih_relabel"]["data_aug"],
            views=config["nih_relabel"]["views"])
        datas.append(dataset)
        datas_names.append("nih_relabel")

    if "nih" in dataset_name:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=config["nih"]["imgpath"],
            transform=config["nih"]["transform"],
            data_aug=config["nih"]["data_aug"],
            views=config["nih"]["views"])
        datas.append(dataset)
        datas_names.append("nih")

    if "pc" in dataset_name:
        dataset = xrv.datasets.PC_Dataset(imgpath=config["pc"]["imgpath"],
                                          transform=config["pc"]["transform"],
                                          data_aug=config["pc"]["data_aug"],
                                          views=config["pc"]["views"])
        datas.append(dataset)
        datas_names.append("pc")

    if "chex" in dataset_name:
        dataset = xrv.datasets.CheX_Dataset(
            imgpath=config["chex"]["imgpath"],
            csvpath=config["chex"]["csvpath"],
            transform=config["chex"]["transform"],
            data_aug=config["chex"]["data_aug"],
            views=config["chex"]["views"])
        datas.append(dataset)
        datas_names.append("chex")

    if "google" in dataset_name:
        dataset = xrv.datasets.NIH_Google_Dataset(
            imgpath=config["google"]["imgpath"],
            transform=config["google"]["transform"],
            data_aug=config["google"]["data_aug"],
            views=config["google"]["views"])
        datas.append(dataset)
        datas_names.append("google")

    if "mimic_ch" in dataset_name:
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=config["mimic_ch"]["imgpath"],
            csvpath=config["mimic_ch"]["csvpath"],
            metacsvpath=config["mimic_ch"]["metacsvpath"],
            transform=config["mimic_ch"]["transform"],
            data_aug=config["mimic_ch"]["data_aug"],
            views=config["mimic_ch"]["views"])
        datas.append(dataset)
        datas_names.append("mimic_ch")

    if "mimic_nb" in dataset_name:
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=config["mimic_nb"]["imgpath"],
            csvpath=config["mimic_nb"]["csvpath"],
            metacsvpath=config["mimic_nb"]["metacsvpath"],
            transform=config["mimic_nb"]["transform"],
            data_aug=config["mimic_nb"]["data_aug"],
            views=config["mimic_nb"]["views"])
        datas.append(dataset)
        datas_names.append("mimic_nb")

    if "openi" in dataset_name:
        dataset = xrv.datasets.Openi_Dataset(
            imgpath=config["openi"]["imgpath"],
            transform=config["openi"]["transform"],
            data_aug=config["openi"]["data_aug"])
        datas.append(dataset)
        datas_names.append("openi")

    if "kaggle" in dataset_name:
        dataset = xrv.datasets.Kaggle_Dataset(
            imgpath=config["kaggle"]["imgpath"],
            transform=config["kaggle"]["transform"],
            data_aug=config["kaggle"]["data_aug"],
            views=config["kaggle"]["views"])
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

    train_loader, valid_loader = parse_dataset_train_val(train_dataset, config)

    # Setting the seed
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("train_dataset.labels.shape", train_dataset.labels.shape)
    print("test_dataset.labels.shape", test_dataset.labels.shape)

    # load model
    model = torch.load(weights_filename, map_location='cpu')

    if config["cuda"]:
        model = model.cuda()

    print(model)

    results = train_utils.valid_test_epoch("valid",
                                           0,
                                           model,
                                           "cuda",
                                           valid_loader,
                                           torch.nn.BCEWithLogitsLoss(),
                                           limit=99999999)

    pickle.dump(results, open(pkl_file, "bw"))
    print(pkl_file)

    print("Done")


def parse_dataset_train_val(dataset, config):
    # Dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    torch.manual_seed(config["seed"])
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size])

    #disable data aug
    valid_dataset.data_aug = None

    # fix labels
    train_dataset.labels = dataset.labels[train_dataset.indices]
    valid_dataset.labels = dataset.labels[valid_dataset.indices]

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["batch_size"],
                                               shuffle=config["shuffle"],
                                               num_workers=config["threads"],
                                               pin_memory=config["cuda"])
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=config["batch_size"],
                                               shuffle=config["shuffle"],
                                               num_workers=config["threads"],
                                               pin_memory=config["cuda"])

    return train_loader, valid_loader


def main(args=None):

    parser = get_parser()
    args_config = parser.parse_args(args)
    config = config_data(model_config=model_config, parser_config=args_config)

    model_name = [
        "all", "all_cropped", "all_cropped_v1", "all_cropped_relabelled-nih",
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
        "nih", "nih_relabel", "pc", "chex", "kaggle", "mimic_nb", "mimic_ch",
        "google", "chex-nih-mimic_ch-pc-google-kaggle",
        "chex-nih-mimic_ch-pc-google-kaggle-mimic_nb",
        "chex-nih_relabel-mimic_ch-pc-google-kaggle",
        "chex-nih_relabel-mimic_ch-pc-google-kaggle-mimic_nb"
    ]

    for count, model in enumerate(model_name):

        for dataset in dataset_name:

            #weight = weight_name[count]
            weight_name = basename(config[model]["weights_url"])
            weight_base = join("/raid/COVID19/models/models_rad_findings",
                               config[model]["base"], "")

            weight = {"base": weight_base, "name": weight_name}

            process_validation(model, weight, dataset, config)


if __name__ == "__main__":

    main()
