from modelvshuman import  Evaluate
from modelvshuman import constants as c
from torchvision import models
from modelvshuman import constants as c
from modelvshuman.plotting.colors import *
from modelvshuman.plotting.decision_makers import DecisionMaker
import os

import torch
from torchvision import models
import torch.nn as nn
from modelvshuman.models.registry import register_model, register_model_any
from modelvshuman.models.wrappers.pytorch import PytorchModel, PyContrastPytorchModel, ClipPytorchModel, \
    ViTPytorchModel, EfficientNetPytorchModel, SwagPytorchModel


from torch.hub import load_state_dict_from_url
from torchvision.models import resnet50

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import List
from os.path import join as pjoin

from . import plotting_helper as ph
from . import constants as consts
from . import analyses as a

@dataclass
class Experiment:
    """
    Experiment parameters
    """
    plotting_conditions: List = field(default_factory=list)
    xlabel: str = 'Condition'
    data_conditions: List = field(default_factory=list)

    def __post_init__(self):
        assert len(self.plotting_conditions) == len(self.data_conditions), \
            "Length of plotting conditions " + str(self.plotting_conditions) + \
            " and data conditions " + str(self.data_conditions) + " must be same"


colour_experiment = Experiment(data_conditions=["cr", "bw"],
                              plotting_conditions=["colour", "greyscale"],
                              xlabel="Colour")

contrast_experiment = Experiment(data_conditions=["c100", "c50", "c30", "c15", "c10", "c05", "c03", "c01"],
                                 plotting_conditions=["100", "50", "30", "15", "10", "5", "3", "1"],
                                 xlabel="Contrast in percent")

high_pass_experiment = Experiment(data_conditions=["inf", "3.0", "1.5", "1.0", "0.7", "0.55", "0.45", "0.4"],
                                  plotting_conditions=["inf", "3.0", "1.5", "1.0", ".7", ".55", ".45", ".4"],
                                  xlabel="Filter standard deviation")

low_pass_experiment = Experiment(data_conditions=["0", "1", "3", "5", "7", "10", "15", "40"],
                                 plotting_conditions=["0", "1", "3", "5", "7", "10", "15", "40"],
                                 xlabel="Filter standard deviation")

phase_scrambling_experiment = Experiment(data_conditions=["0", "30", "60", "90", "120", "150", "180"],
                                         plotting_conditions=["0", "30", "60", "90", "120", "150", "180"],
                                         xlabel="Phase noise width [$^\circ$]")

power_equalisation_experiment = Experiment(data_conditions=["0", "pow"],
                                           plotting_conditions=["original", "equalised"],
                                           xlabel="Power spectrum")

false_colour_experiment = Experiment(data_conditions=["True", "False"],
                                    plotting_conditions=["true", "opponent"],
                                    xlabel="Colour")

rotation_experiment = Experiment(data_conditions=["0", "90", "180", "270"],
                                 plotting_conditions=["0", "90", "180", "270"],
                                 xlabel="Rotation angle [$^\circ$]")

_eidolon_plotting_conditions = ["0", "1", "2", "3", "4", "5", "6", "7"]
_eidolon_xlabel = "$\mathregular{{Log}_2}$ of 'reach' parameter"

eidolonI_experiment = Experiment(data_conditions=["1-10-10", "2-10-10", "4-10-10", "8-10-10",
                                                  "16-10-10", "32-10-10", "64-10-10", "128-10-10"],
                                 plotting_conditions=_eidolon_plotting_conditions.copy(),
                                 xlabel=_eidolon_xlabel)

eidolonII_experiment = Experiment(data_conditions=["1-3-10", "2-3-10", "4-3-10", "8-3-10",
                                                   "16-3-10", "32-3-10", "64-3-10", "128-3-10"],
                                  plotting_conditions=_eidolon_plotting_conditions.copy(),
                                  xlabel=_eidolon_xlabel)

eidolonIII_experiment = Experiment(data_conditions=["1-0-10", "2-0-10", "4-0-10", "8-0-10",
                                                    "16-0-10", "32-0-10", "64-0-10", "128-0-10"],
                                   plotting_conditions=_eidolon_plotting_conditions.copy(),
                                   xlabel=_eidolon_xlabel)

uniform_noise_experiment = Experiment(data_conditions=["0.0", "0.03", "0.05", "0.1", "0.2", "0.35", "0.6", "0.9"],
                                      plotting_conditions=["0.0", ".03", ".05", ".1", ".2", ".35", ".6", ".9"],
                                      xlabel="Uniform noise width")


@dataclass
class DatasetExperiments:
    name: str
    experiments: [Experiment]


def get_experiments(dataset_names):
    datasets = []
    for name in dataset_names:
        name_for_experiment = name.replace("-", "_")
        if f"{name_for_experiment}_experiment" in globals():
            experiments = eval(f"{name_for_experiment}_experiment")
            experiments.name = name
            datasets.append(DatasetExperiments(name=name, experiments=[experiments]))
        else:
            datasets.append(DatasetExperiments(name=name, experiments=[]))
    return datasets


def get_dataset_names(plot_type):
    """Given plot_type, return suitable dataset(s).

    In this regard, 'suitable' means:
    - valid plot_type <-> dataset combination
    - data is available
    """

    dataset_names = []
    dataset_candidates = consts.PLOT_TYPE_TO_DATASET_MAPPING[plot_type]

    for candidate in dataset_candidates:
        if os.path.exists(pjoin(consts.RAW_DATA_DIR, candidate)):
            dataset_names.append(candidate)

    if len(dataset_names) == 0:
        raise ValueError("No data found for the specified plot_types.")

    return dataset_names


def plot(plot_types,
         plotting_definition,
         dataset_names=None,
         figure_directory_name="example-figures",
         crop_PDFs=True,
         *args, **kwargs):

        current_dataset_names = get_dataset_names(plot_types[0])
        datasets = get_experiments(current_dataset_names)
        result_dir = os.path.join(consts.FIGURE_DIR, figure_directory_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return plot_shape_bias_matrixplot(datasets=datasets,
                                    decision_maker_fun=plotting_definition,
                                    result_dir=result_dir)

def plot_shape_bias_matrixplot(datasets,
                               decision_maker_fun,
                               result_dir,
                               analysis=a.ShapeBias(),
                               order_by='humans'):
    assert len(datasets) == 1
    ds = datasets[0]
    assert ds.name == "cue-conflict"
    df = ph.get_experimental_data(ds)
    classes = df["category"].unique()
    num_classes = len(classes)

    result=[]
    # plot average shapebias + scatter points
    for dmaker in decision_maker_fun(df):
        df_selection = df.loc[(df["subj"].isin(dmaker.decision_makers))]
        result_df = analysis.analysis(df=df_selection)
        avg = 1 - result_df['shape-bias']
        for cl in classes:
            df_class_selection = df_selection.query("category == '{}'".format(cl))
            result.append((cl,analysis.analysis(df=df_class_selection)['shape-bias']))
    return result


def plotting_definition_new(df):

    decision_makers = []
    decision_makers.append(DecisionMaker(name_pattern="anyModel",
                           color=rgb(80, 220, 60), marker="o", df=df,
                           plotting_name="anyModel"))
    # decision_makers.append(DecisionMaker(name_pattern="subject-*",
    #                        color=rgb(165, 30, 55), marker="D", df=df,
    #                        plotting_name="humans"))
    return decision_makers


def run_evaluation(model, modelnames="anyModel"):

    model = PytorchModel(model, modelnames, None)
    register_model_any("pytorch_any", modelnames, model)

    test_models = [modelnames]
    datasets = ["cue-conflict"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(test_models, datasets, **params)

    return run_plotting()


def run_plotting():
    plot_types=["shape-bias"]
    plotting_def = plotting_definition_new
    figure_dirname = "example-figures/"

    return plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)



if __name__ == "__main__":
    modelnames="anyModel"
    model_ckpt = f"/home/ylz1122/ffcv-imagenet-train/scripts/alexnet/train_results/alexnet_configs/alexnet_5layers_32_epochs_test_finetune_alex_2layer-original/06eec4be-88e5-43ef-8f52-6d9f827d8805/weights_ep_{ep}.pt"
    ep = 5
    vgg16 = models.vgg16(pretrained=True)
    # 1. evaluate models on out-of-distribution datasets
    result = run_evaluation(modelnames,vgg16)
    print(result)
