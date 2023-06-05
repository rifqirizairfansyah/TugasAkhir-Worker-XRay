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
    
    out_op_threshs = op_norm(output_sigmoid, op_threshs)
    
    out_lung_opacity = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Lung Opacity")]
    out_consolidation = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Consolidation")]
    out_pneumonia = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Pneumonia")]
    out_infiltration = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Infiltration")]
    out_atelectasis = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Atelectasis")]
    out_pneumothorax = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Pneumothorax")]
    out_edema = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Edema")]
    out_emphysema = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Emphysema")]
    out_fibrosis = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Fibrosis")]
    out_effusion = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Effusion")]
    out_pleural_thickening = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Pleural_Thickening")]
    out_cardiomegaly = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Cardiomegaly")]
    out_nodule = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Nodule")]
    out_mass = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Mass")]
    out_hernia = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Hernia")]
    out_lung_lesion = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Lung Lesion")]
    out_fracture = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Fracture")]
    out_enlarged_cardiomediastinum = output_sigmoid[
        0, xrv.datasets.default_pathologies.index("Enlarged Cardiomediastinum")]
    
    # output of op threshs
    out_lung_opacity_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Lung Opacity")]
    out_consolidation_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Consolidation")]
    out_pneumonia_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Pneumonia")]
    out_infiltration_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Infiltration")]
    out_atelectasis_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Atelectasis")]
    out_pneumothorax_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Pneumothorax")]
    out_edema_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Edema")]
    out_emphysema_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Emphysema")]
    out_fibrosis_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Fibrosis")]
    out_effusion_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Effusion")]
    out_pleural_thickening_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Pleural_Thickening")]
    out_cardiomegaly_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Cardiomegaly")]
    out_nodule_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Nodule")]
    out_mass_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Mass")]
    out_hernia_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Hernia")]
    out_lung_lesion_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Lung Lesion")]
    out_fracture_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Fracture")]
    out_enlarged_cardiomediastinum_op = out_op_threshs[
        0, xrv.datasets.default_pathologies.index("Enlarged Cardiomediastinum")]
    
    
    
    # output severity : area of opacity and degree of opacity
    area_op = outputs["geographic_extent"]
    deg_op = outputs["opacity"]
    
    abnormalitas_citra = area_op *100 / 8
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
        "atelectasis" : out_atelectasis.cpu().detach().numpy(),
        "pneumothorax" : out_pneumothorax.cpu().detach().numpy(),
        "edema" : out_edema.cpu().detach().numpy(),
        "emphysema" : out_emphysema.cpu().detach().numpy(),
        "fibrosis" : out_fibrosis.cpu().detach().numpy(),
        "effusion" : out_effusion.cpu().detach().numpy(),
        "pleural_thickening" : out_pleural_thickening.cpu().detach().numpy(),
        "cardiomegaly" : out_cardiomegaly.cpu().detach().numpy(),
        "nodule" : out_nodule.cpu().detach().numpy(),
        "mass" : out_mass.cpu().detach().numpy(),
        "hernia" : out_hernia.cpu().detach().numpy(),
        "lung_lesion" : out_lung_lesion.cpu().detach().numpy(),
        "fracture" : out_fracture.cpu().detach().numpy(),
        "enlarged_cardiomediastinum" : out_enlarged_cardiomediastinum.cpu().detach().numpy(),
        "lung_opacity_op" : out_lung_opacity_op.cpu().detach().numpy(),
        "consolidation_op" : out_consolidation_op.cpu().detach().numpy(),
        "pneumonia_op" : out_pneumonia_op.cpu().detach().numpy(),
        "infiltration_op" : out_infiltration_op.cpu().detach().numpy(),
        "atelectasis_op" : out_atelectasis_op.cpu().detach().numpy(),
        "pneumothorax_op" : out_pneumothorax_op.cpu().detach().numpy(),
        "edema_op" : out_edema_op.cpu().detach().numpy(),
        "emphysema_op" : out_emphysema_op.cpu().detach().numpy(),
        "fibrosis_op" : out_fibrosis_op.cpu().detach().numpy(),
        "effusion_op" : out_effusion_op.cpu().detach().numpy(),
        "pleural_thickening_op" : out_pleural_thickening_op.cpu().detach().numpy(),
        "cardiomegaly_op" : out_cardiomegaly_op.cpu().detach().numpy(),
        "nodule_op" : out_nodule_op.cpu().detach().numpy(),
        "mass_op" : out_mass_op.cpu().detach().numpy(),
        "hernia_op" : out_hernia_op.cpu().detach().numpy(),
        "lung_lesion_op" : out_lung_lesion_op.cpu().detach().numpy(),
        "fracture_op" : out_fracture_op.cpu().detach().numpy(),
        "enlarged_cardiomediastinum_op" : out_enlarged_cardiomediastinum_op.cpu().detach().numpy(),
        "area_opacity" : area_op.cpu().detach().numpy(),
        "degree_opacity" : deg_op.cpu().detach().numpy(), 
        "abnormalitas_citra" : abnormalitas_citra.cpu().detach().numpy(),
        "saliency_path" : saliency_path
    }
    
    return result



def op_norm(outputs, op_threshs):
    outputs_new = torch.zeros(outputs.shape, device=outputs.device)
    for i in range(len(outputs)):
        for t in range(len(outputs[0])):
            if (outputs[i,t]<op_threshs[t]):
                outputs_new[i,t] = outputs[i,t]/(op_threshs[t]*2) 
            else:
                outputs_new[i,t] = 1-((1-outputs[i,t])/((1-(op_threshs[t]))*2)) 
            
    return outputs_new


def save_dataframe_toexcel(list_source, list_patient_id, list_file_name, list_status, list_lung_opacity, list_consolidation, list_pneumonia, list_infiltration, list_atelectasis, list_pneumothorax, list_edema, list_emphysema, list_fibrosis, list_effusion, list_pleural_thickening, list_cardiomegaly, list_nodule, list_mass, list_hernia, list_lung_lesion, list_fracture, list_enlarged_cardiomediastinum, list_lung_opacity_op, list_consolidation_op, list_pneumonia_op, list_infiltration_op, list_atelectasis_op, list_pneumothorax_op, list_edema_op, list_emphysema_op, list_fibrosis_op, list_effusion_op, list_pleural_thickening_op, list_cardiomegaly_op, list_nodule_op, list_mass_op, list_hernia_op, list_lung_lesion_op, list_fracture_op, list_enlarged_cardiomediastinum_op, list_area_opacity, list_degree_opacity, list_abnormalitas_citra, list_saliency_path, path_data_frame):
    
    inference_result = {
            "source" : list_source,
            "patient_id" : list_patient_id ,
            "file_name" : list_file_name,
            "status" : list_status,
            "lung_opacity" : list_lung_opacity,
            "consolidation" : list_consolidation,
            "pneumonia" : list_pneumonia,
            "infiltration" : list_infiltration,
            "atelectasis" : list_atelectasis,
            "pneumothorax" : list_pneumothorax,
            "edema" : list_edema,
            "emphysema" : list_emphysema,
            "fibrosis" : list_fibrosis,
            "effusion" : list_effusion,
            "pleural_thickening" : list_pleural_thickening,
            "cardiomegaly" : list_cardiomegaly,
            "nodule" : list_nodule,
            "mass" : list_mass,
            "hernia" : list_hernia,
            "lung_lesion" : list_lung_lesion,
            "fracture" : list_fracture,
            "enlarged_cardiomediastinum" : list_enlarged_cardiomediastinum,
            "lung_opacity_op" : list_lung_opacity_op,
            "consolidation_op" : list_consolidation_op,
            "pneumonia_op" : list_pneumonia_op,
            "infiltration_op" : list_infiltration_op,
            "atelectasis_op" : list_atelectasis_op,
            "pneumothorax_op" : list_pneumothorax_op,
            "edema_op" : list_edema_op,
            "emphysema_op" : list_emphysema_op,
            "fibrosis_op" : list_fibrosis_op,
            "effusion_op" : list_effusion_op,
            "pleural_thickening_op" : list_pleural_thickening_op,
            "cardiomegaly_op" : list_cardiomegaly_op,
            "nodule_op" : list_nodule_op,
            "mass_op" : list_mass_op,
            "hernia_op" : list_hernia_op,
            "lung_lesion_op" : list_lung_lesion_op,
            "fracture_op" : list_fracture_op,
            "enlarged_cardiomediastinum_op" : list_enlarged_cardiomediastinum_op,
            'area_opacity' : list_area_opacity,
            'degree_opacity' : list_degree_opacity,
            "abnormalitas_citra" : list_abnormalitas_citra,
            'saliency_path' : list_saliency_path
            } 

    df = pd.DataFrame(inference_result, columns = [ 
            "source",
            "patient_id" ,
            "file_name" ,
            "status" ,
            "lung_opacity" ,
            "consolidation" ,
            "pneumonia" ,
            "infiltration" ,
            "atelectasis" ,
            "pneumothorax" ,
            "edema" ,
            "emphysema" ,
            "fibrosis" ,
            "effusion" ,
            "pleural_thickening" ,
            "cardiomegaly" ,
            "nodule" ,
            "mass" ,
            "hernia" ,
            "lung_lesion" ,
            "fracture",
            "enlarged_cardiomediastinum" ,
            "lung_opacity_op" ,
            "consolidation_op" ,
            "pneumonia_op" ,
            "infiltration_op" ,
            "atelectasis_op" ,
            "pneumothorax_op" ,
            "edema_op" ,
            "emphysema_op" ,
            "fibrosis_op" ,
            "effusion_op" ,
            "pleural_thickening_op" ,
            "cardiomegaly_op" ,
            "nodule_op" ,
            "mass_op" ,
            "hernia_op" ,
            "lung_lesion_op" ,
            "fracture_op",
            "enlarged_cardiomediastinum_op" ,
            'area_opacity',
            'degree_opacity' ,
            "abnormalitas_citra",
            'saliency_path' ])   
    df.to_excel (path_data_frame, index = False, header=True)


path_source = "/raid/COVID19/dataset_evaluasi/model_0/cropped_xlsor/"
path_output = "/raid/COVID19/dataset_evaluasi/model_1_2/saliency_path_1_op/"
path_data_frame = "/raid/COVID19/dataset_evaluasi/model_1_2/result_model_0_1_2_new_op.xlsx"

op_threshs = [0.07422872, 0.038290843, 0.09814756, 0.0098118475, 0.023601074, 0.0022490358, 0.010060724, 0.103246614, 0.056810737, 0.026791653, 0.050318155, 0.023985857, 0.01939503, 0.042889766, 0.053369623, 0.035975814, 0.20204692, 0.05015312]


list_source = []
list_patient_id = []
list_file_name = []
list_status = []

list_lung_opacity = []
list_consolidation = []
list_pneumonia = []
list_infiltration = []
list_atelectasis= []
list_pneumothorax= []
list_edema= []
list_emphysema= []
list_fibrosis= []
list_effusion= []
list_pleural_thickening= []
list_cardiomegaly= []
list_nodule= []
list_mass= []
list_hernia= []
list_lung_lesion= []
list_fracture= []
list_enlarged_cardiomediastinum= []

list_lung_opacity_op = []
list_consolidation_op = []
list_pneumonia_op = []
list_infiltration_op = []
list_atelectasis_op= []
list_pneumothorax_op= []
list_edema_op= []
list_emphysema_op= []
list_fibrosis_op= []
list_effusion_op= []
list_pleural_thickening_op= []
list_cardiomegaly_op= []
list_nodule_op= []
list_mass_op= []
list_hernia_op= []
list_lung_lesion_op= []
list_fracture_op= []
list_enlarged_cardiomediastinum_op= []

list_area_opacity = []
list_degree_opacity = [] 
list_abnormalitas_citra =[]

list_saliency_path = []


def main():
    
    
    model_pneumonia_severity = PneumoniaSeverityNet()
    
    folder_source = [f for f in listdir(path_source) if isdir(join(path_source, f))]

    for folder in folder_source :

        source = folder

        status_source = join(path_source,folder)

        folder_status = [f for f in listdir(status_source) if isdir(join(status_source, f))]

        for folder in folder_status :

            status = folder

            patient_source = join(status_source,folder)

            folder_patient = [f for f in listdir(patient_source) if isdir(join(patient_source, f))]

            for folder in folder_patient :

                patient_id = folder

                file_source = join(patient_source,folder)

                files = [f for f in listdir(file_source) if isfile(join(file_source, f))]

                for file in files :
                    
                    # path setting and processsing
                    # file path to load image
                    file_name = file
                    split_name = splitext(file_name)
                    file_name = split_name[0]
                    file_path = join(file_source, file)

                    print(source, patient_id, file_name, status, file_path)
                    # saliency path
                    saliency_dir = os.path.join(path_output, source, status, patient_id)
                    
                    if (os.path.isdir(saliency_dir)==False) : 
                        os.makedirs(saliency_dir)
    
                    saliency_file = file_name + "_result"+ ".png"  
                    saliency_path = os.path.join(saliency_dir, saliency_file)
                    
                    # process adn inference image
                    image_cv = cv2.imread(file_path)
                    image_preprocess = preprocessing_image(image_cv)
                    result = inference_all(image_preprocess, model_pneumonia_severity, saliency_path)
                    
       
                    lung_opacity = float(result['lung_opacity'])
                    consolidation = float(result['consolidation'])
                    pneumonia = float(result['pneumonia'])
                    infiltration = float(result['infiltration'])
                    atelectasis = float(result['atelectasis'])
                    pneumothorax = float(result['pneumothorax'])
                    edema = float(result['edema'])
                    emphysema = float(result['emphysema'])
                    fibrosis = float(result['fibrosis'])
                    effusion = float(result['effusion'])
                    pleural_thickening = float(result['pleural_thickening'])
                    cardiomegaly = float(result['cardiomegaly'])
                    nodule = float(result['nodule'])
                    mass = float(result['mass'])
                    hernia = float(result['hernia'])
                    lung_lesion = float(result['lung_lesion'])
                    fracture = float(result['fracture'])
                    enlarged_cardiomediastinum = float(result['enlarged_cardiomediastinum'])


                    lung_opacity_op = float(result['lung_opacity_op'])
                    consolidation_op = float(result['consolidation_op'])
                    pneumonia_op = float(result['pneumonia_op'])
                    infiltration_op = float(result['infiltration_op'])
                    atelectasis_op = float(result['atelectasis_op'])
                    pneumothorax_op = float(result['pneumothorax_op'])
                    edema_op = float(result['edema_op'])
                    emphysema_op = float(result['emphysema_op'])
                    fibrosis_op = float(result['fibrosis_op'])
                    effusion_op = float(result['effusion_op'])
                    pleural_thickening_op = float(result['pleural_thickening_op'])
                    cardiomegaly_op = float(result['cardiomegaly_op'])
                    nodule_op = float(result['nodule_op'])
                    mass_op = float(result['mass_op'])
                    hernia_op = float(result['hernia_op'])
                    lung_lesion_op = float(result['lung_lesion_op'])
                    fracture_op = float(result['fracture_op'])
                    enlarged_cardiomediastinum_op = float(result['enlarged_cardiomediastinum_op'])

                    area_opacity = float(result['area_opacity'])
                    degree_opacity = float(result['degree_opacity'])
                    abnormalitas_citra = float(result['abnormalitas_citra'])

                    
                    list_source.append(source)
                    list_patient_id.append(patient_id)
                    list_file_name.append(file_name)
                    list_status.append(status)

                    list_lung_opacity.append(lung_opacity)
                    list_consolidation.append(consolidation)
                    list_pneumonia.append(pneumonia)
                    list_infiltration.append(infiltration)
                    list_atelectasis.append(atelectasis)
                    list_pneumothorax.append(pneumothorax)
                    list_edema.append(edema)
                    list_emphysema.append(emphysema)
                    list_fibrosis.append(fibrosis)
                    list_effusion.append(effusion)
                    list_pleural_thickening.append(pleural_thickening)
                    list_cardiomegaly.append(cardiomegaly)
                    list_nodule.append(nodule)
                    list_mass.append(mass)
                    list_hernia.append(hernia)
                    list_lung_lesion.append(lung_lesion)
                    list_fracture.append(fracture)
                    list_enlarged_cardiomediastinum.append(enlarged_cardiomediastinum)

                    list_lung_opacity_op.append(lung_opacity_op)
                    list_consolidation_op.append(consolidation_op)
                    list_pneumonia_op.append(pneumonia_op)
                    list_infiltration_op.append(infiltration_op)
                    list_atelectasis_op.append(atelectasis_op)
                    list_pneumothorax_op.append(pneumothorax_op)
                    list_edema_op.append(edema_op)
                    list_emphysema_op.append(emphysema_op)
                    list_fibrosis_op.append(fibrosis_op)
                    list_effusion_op.append(effusion_op)
                    list_pleural_thickening_op.append(pleural_thickening_op)
                    list_cardiomegaly_op.append(cardiomegaly_op)
                    list_nodule_op.append(nodule_op)
                    list_mass_op.append(mass_op)
                    list_hernia_op.append(hernia_op)
                    list_lung_lesion_op.append(lung_lesion_op)
                    list_fracture_op.append(fracture_op)
                    list_enlarged_cardiomediastinum_op.append(enlarged_cardiomediastinum_op)

                    list_area_opacity.append(area_opacity)
                    list_degree_opacity.append(degree_opacity) 
                    list_abnormalitas_citra.append(abnormalitas_citra)

                    list_saliency_path.append(saliency_path)
                    
                    #save_dataframe_toexcel(list_source, list_patient_id, list_file_name, list_status, list_lung_opacity, list_consolidation, list_pneumonia, list_infiltration, list_atelectasis, list_pneumothorax, list_edema, list_emphysema, list_fibrosis, list_effusion, list_pleural_thickening, list_cardiomegaly, list_nodule, list_mass, list_hernia, list_lung_lesion, list_fracture, list_enlarged_cardiomediastinum, list_area_opacity, list_degree_opacity, list_abnormalitas_citra, list_saliency_path, path_data_frame)
        
        
    save_dataframe_toexcel(list_source, list_patient_id, list_file_name, list_status, list_lung_opacity, list_consolidation, list_pneumonia, list_infiltration, list_atelectasis, list_pneumothorax, list_edema, list_emphysema, list_fibrosis, list_effusion, list_pleural_thickening, list_cardiomegaly, list_nodule, list_mass, list_hernia, list_lung_lesion, list_fracture, list_enlarged_cardiomediastinum, list_lung_opacity_op, list_consolidation_op, list_pneumonia_op, list_infiltration_op, list_atelectasis_op, list_pneumothorax_op, list_edema_op, list_emphysema_op, list_fibrosis_op, list_effusion_op, list_pleural_thickening_op, list_cardiomegaly_op, list_nodule_op, list_mass_op, list_hernia_op, list_lung_lesion_op, list_fracture_op, list_enlarged_cardiomediastinum_op, list_area_opacity, list_degree_opacity, list_abnormalitas_citra, list_saliency_path, path_data_frame)
    
if __name__ == "__main__":
    main()