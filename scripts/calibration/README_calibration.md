# Calibration of Radiological Findings Model

## Configuration 
```
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
```
configurations :
1. transform_coonfig : for type of dataset used in calibration models such as standard (center crop and resize 224x224) and cropped (cropped lung segmentation and resize 224x224)
2. model_config : locate .pt file as saved model for each developed models contained weights_url as filename of .pt model and base as based folder of saved model
3. weigth_name : call location of weight of models
4. dataset_name : dataset combination used for calibration 

## Run script and notebook

```
python3 model_calibrate.py
```

then, run the jupyter notebook to calibrate model from .pkl files after executing model_calibrate.py script,

```
jupyter notebook --allow-root --no-browser --port 6664 --NotebookApp.token='' --ip 0.0.0.0 
```
## Calibration Result

**all_cropped**
```
results_chex-nih-mimic_nb-mimic_ch-pc-google-openi-kaggle-densenet121-model_2_cropped-best_chex-nih-mimic_ch-pc-google-openi-kaggle-mimic_nb.pkl

all_threshs :
[0.08653892, 0.017242203, 0.061408978, 0.006965817, 0.007230344, 0.008352524, 0.0052154628, 0.09193552, 0.035084374, 0.020144384, 0.0809918, 0.032811552, 0.0071497215, 0.014541325, 0.019512547, 0.031239916, 0.11984462, 0.00936549]

all_ppv80 :
[0.9937518, 0.99484503, 0.57375365, 0.9282644, 0.9529527, 0.5661315, 0.4997312, 0.91090864, 0.99205655, 0.8359465, 0.9208461, 0.91701555, 0.9997881, 0.8658175, 0.99993837, 0.99887985, 0.8762074, 0.9988675]

thres[ppv80_thres] : 5.0772295
```

**all_cropped_v1**
```
results_chex-nih-mimic_ch-pc-google-openi-kaggle-densenet121-cropped-best_chex-nih-mimic_ch-pc-google-openi-kaggle.pkl

all_threshs :
[0.04718963, 0.015039951, 0.07342019, 0.0051146024, 0.007409899, 0.008317836, 0.004983357, 0.04215132, 0.030943004, 0.031830408, 0.039926834, 0.031946592, 0.0096772835, 0.016858106, 0.055931136, 0.046968885, 0.15709805, 0.015543708]

all_ppv80 :
[0.8200604, 0.8610378, 0.64609003, 0.7353858, 0.96263915, 0.6738628, 0.8235256, 0.5164257, 0.9624524, 0.85824424, 0.7721861, 0.91494757, 0.9859875, 0.4577153, 0.9642816, 0.9072195, 0.6847318, 0.9756423]

thres[ppv80_thres] : 1.5170751
```


**all_cropped_relabeled-nih**
```
results_chex-nih-mimic_nb-mimic_ch-pc-google-openi-kaggle-densenet121-model_2_cropped_relabel-nih-best_chex-nih_relabel-mimic_ch-pc-google-openi-kaggle-mimic_nb.pkl

all_threshs :
[0.043316294, 0.01043874, 0.07821479, 0.0030074122, 0.0113249775, 0.010188991, 0.0059004985, 0.05317818, 0.037837196, 0.0050458475, 0.09594208, 0.040771976, 0.007820424, 0.015169545, 0.006620609, 0.029966863, 0.08549373, 0.006000353]

all_ppv80 :
[0.999503, 0.99491274, 0.9934208, 0.96971613, 0.9995702, 0.8879771, 0.9983138, 0.890265, 0.9672664, 0.9856338, 0.98278266, 0.99870896, 0.9999771, 0.9739644, 0.9996896, 0.9940188, 0.9450582, 0.9978744]

thres[ppv80_thres] : 7.644227
```

**all_cropped_relabelled-nih_v1**
```
results_chex-nih-mimic_ch-pc-google-openi-kaggle-densenet121-cropped_relabel-nih-best_chex-nih_relabel-mimic_ch-pc-google-openi-kaggle.pkl

all_threshs :
[0.06420791, 0.011237458, 0.046035413, 0.0052339705, 0.011177873, 0.0024271521, 0.0063031274, 0.06383512, 0.0338762, 0.02408131, 0.06494272, 0.037538163, 0.0054588974, 0.026531212, 0.044195093, 0.0327385, 0.17920227, 0.024809295]

all_ppv80 :
[0.93297035, 0.9559677, 0.80296314, 0.8045187, 0.94467974, 0.481314, 0.9366123, 0.7138835, 0.96114165, 0.7748556, 0.8895693, 0.74475545, 0.9937219, 0.9175519, 0.9321702, 0.84740615, 0.65827876, 0.93292886]

thres[ppv80_thres] : 2.6337724
```
