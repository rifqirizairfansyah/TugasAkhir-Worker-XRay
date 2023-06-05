#!/usr/bin/env python
# coding: utf-8
import os, sys
from os import listdir
from os.path import isfile, join
sys.path.insert(0, "..")
sys.path.insert(0, "/workspace/update/torchxrayvision")
sys.path.insert(0, "/workspace/update/torchxrayvision/torchxrayvision")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage
import pprint
import cv2
# Torch X Ray Vision Lib
import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
import skimage, skimage.filters
import torchxrayvision as xrv

# Data Frame Lib
import pandas as pd

def full_frame(width=None, height=None):
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)

def preprocessing_image(img) :

    img = xrv.datasets.normalize(img, 255)

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]
    transform = torchvision.transforms.Compose([xrv.datasets.XRayResizer(224)])

    img = transform(img)
    
    return img


class PneumoniaSeverityNet(torch.nn.Module):
    def __init__(self):
        super(PneumoniaSeverityNet, self).__init__()
        self.model = xrv.models.DenseNet(weights="all")
        self.model.op_threshs = None
        self.theta_bias_geographic_extent = torch.from_numpy(
            np.asarray((0.8705248236656189, 3.4137437)))
        self.theta_bias_opacity = torch.from_numpy(
            np.asarray((0.5484423041343689, 2.5535977)))

    def forward(self, x):
        preds_all = self.model(x)
        preds = preds_all[0,
                      xrv.datasets.default_pathologies.index("Lung Opacity")]
        geographic_extent = preds * self.theta_bias_geographic_extent[
            0] + self.theta_bias_geographic_extent[1]
        opacity = preds * self.theta_bias_opacity[0] + self.theta_bias_opacity[
            1]
        geographic_extent = torch.clamp(geographic_extent, 0, 8)
        opacity = torch.clamp(opacity, 0, 6)
        return {"radiological_findings" : preds_all, "geographic_extent": geographic_extent, "opacity": opacity}

def saving_saliency_result(img, blurred, saliency_path) :
    full_frame()

    plt.imshow(img[0][0].cpu().detach(), cmap="gray", aspect='auto')
    plt.imshow(blurred, alpha=0.3)
    plt.savefig(saliency_path)
    
    return True, saliency_path

def inference_all(img, model, saliency_path):

    # running on gpu
    with torch.no_grad():
        # define image annd model in gpu
        img = torch.from_numpy(img).unsqueeze(0)

    img = img.cuda()
    model = model.cuda()

    img = img.requires_grad_()

    outputs = model(img)

    # output radiological findings : Lung Opacity, Consolidation, Pneumonia, Infiltration
    output_findings = outputs["radiological_findings"]
    output_sigmoid = torch.sigmoid(output_findings)

    out_lung_opacity = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Lung Opacity")]
    out_consolidation = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Consolidation")]
    out_pneumonia = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Pneumonia")]
    out_infiltration = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Infiltration")]

    # output severity : area of opacity and degree of opacity
    area_op = outputs["geographic_extent"]
    deg_op = outputs["opacity"]

    # output saliency map : heatmap of pneumonia severity
    #area_op_saliency = area_op.requires_grad_()

    grads = torch.autograd.grad(outputs["geographic_extent"],
                                img)[0][0][0]
    grads_num = grads.cpu().detach().numpy()

    grad_tensor = torch.from_numpy(grads_num).float().to("cpu")

    blurred = skimage.filters.gaussian(grad_tensor**2,
                                       sigma=(5, 5),
                                       truncate=3.5)

    saliency_path = saving_saliency_result(img, blurred, saliency_path)

    result = {
        "lung_opacity" : out_lung_opacity.cpu().detach().numpy(),
        "consolidation" : out_consolidation.cpu().detach().numpy(),
        "pneumonia" : out_pneumonia.cpu().detach().numpy(),
        "infiltration" : out_infiltration.cpu().detach().numpy(),
        "area_opacity" : area_op.cpu().detach().numpy(),
        "degree_opacity" : deg_op.cpu().detach().numpy(), 
        "saliency_path" : saliency_path
    }
    
    return result


def save_dataframe_toexcel(list_id_image, list_lung_opacity, list_consolidation, list_pneumonia, list_infiltration, list_area_opacity, list_degree_opacity, list_saliency_path, path_data_frame):
    
    inference_result = {
        'Image ID': list_id_image,
        'Lung Opacity': list_lung_opacity,
        'Consolidation' : list_consolidation,
        'Pneumonia' : list_pneumonia,
        'Infiltration' : list_infiltration,
        'Area Opacity' : list_area_opacity,
        'Degree Opacity' : list_degree_opacity,
        'Saliency Path' : list_saliency_path
    } 

    df = pd.DataFrame(inference_result, columns = ['Image ID', 'Lung Opacity', 'Consolidation', 'Pneumonia', 'Infiltration', 'Area Opacity', 'Degree Opacity','Saliency Path'])
    
    df.to_excel (path_data_frame, index = False, header=True)
    


path_load_image = "/workspace/dataset/dataset_evaluasi/model_0/cropped_unet/"
path_output = "/workspace/dataset/dataset_evaluasi/model_1_2/saliency_path/"
path_data_frame = "/workspace/dataset/dataset_evaluasi/model_1_2/result_model_0_1_2.xlsx"

list_id_image = []
list_lung_opacity = []
list_consolidation = []
list_pneumonia = []
list_infiltration = []
list_area_opacity = []
list_degree_opacity = [] 
list_saliency_path = []


def main():
    
    
    model_pneumonia_severity = PneumoniaSeverityNet()
    
    files = [f for f in listdir(path_load_image) if isfile(join(path_load_image, f))]
    
    # inference each image
    for file in files :
    
        file_path = os.path.join(path_load_image, file) 
        split_name = os.path.splitext(file)
        file_name = split_name[0]
        id_image = file_name.replace('_cropped', '')

        #print(id_image, file_name, file_path)
        image_cv = cv2.imread(file_path)
        
        image_preprocess = preprocessing_image(image_cv)
        
        saliency_path = os.path.join(path_output, file_name + "_result"+ ".png")
        
        print(file_name)
        result = inference_all(image_preprocess, model_pneumonia_severity, saliency_path)
    
        
        lung_opacity = float(result['lung_opacity'])
        consolidation = float(result['consolidation'])
        pneumonia = float(result['pneumonia'])
        infiltration = float(result['infiltration'])
        area_opacity = float(result['area_opacity'])
        degree_opacity = float(result['degree_opacity'])
        
        
        list_id_image.append(id_image)
        list_lung_opacity.append(lung_opacity)
        list_consolidation.append(consolidation)
        list_pneumonia.append(pneumonia)
        list_infiltration.append(infiltration)
        list_area_opacity.append(area_opacity)
        list_degree_opacity.append(degree_opacity) 
        list_saliency_path.append(saliency_path)
        
        
    save_dataframe_toexcel(list_id_image, list_lung_opacity, list_consolidation, list_pneumonia, list_infiltration, list_area_opacity, list_degree_opacity, list_saliency_path, path_data_frame)
    
if __name__ == "__main__":
    main()