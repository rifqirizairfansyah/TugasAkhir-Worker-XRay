model_config = {}

model_config['all'] = {
    "weights_url":
    'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "base": "cohen"
}

model_config['nih'] = {
    "weights_url":
    'nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "base": "cohen"
}

model_config['pc'] = {
    "weights_url":
    'pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "base": "cohen"
}

model_config['chex'] = {
    "weights_url":
    'chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "base": "cohen"
}

model_config['kaggle'] = {
    "weights_url":
    'kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "base": "cohen"
}

model_config['mimic_nb'] = {
    "weights_url":
    'mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "base": "cohen"
}

model_config['mimic_ch'] = {
    "weights_url":
    'mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "base": "cohen"
}

model_config['all_cropped'] = {
    "weights_url":
    'chex-nih-mimic_nb-mimic_ch-pc-google-openi-kaggle-densenet121-model_2_cropped-best.pt',
    "base": "train-cropped"
}

model_config['all_cropped_v1'] = {
    "weights_url":
    'chex-nih-mimic_ch-pc-google-openi-kaggle-densenet121-cropped-best.pt',
    "base": "train-cropped"
}

model_config['all_cropped_relabelled-nih'] = {
    "weights_url":
    'chex-nih-mimic_nb-mimic_ch-pc-google-openi-kaggle-densenet121-model_2_cropped_relabel-nih-best.pt',
    "base": "train-cropped-relabelled-nih"
}

model_config['all_cropped_relabelled-nih_v1'] = {
    "weights_url":
    'chex-nih-mimic_ch-pc-google-openi-kaggle-densenet121-cropped_relabel-nih-best.pt',
    "base": "train-cropped-relabelled-nih"
}

model_config['first'] = {
    "weights_url":
    'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-first-best.pt',
    "base": "train-02-08-2020"
}

model_config['relabelled-nih'] = {
    "weights_url":
    'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-relabelled-nih-best.pt',
    "base": "train-relabelled-nih"
}

model_config['mixed-normal-contour'] = {
    "weights_url":
    'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-mixed-normal-contour-best.pt',
    "base": "train-mixed-lung-contour"
}