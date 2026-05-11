from sklearn import metrics
import numpy as np
import copy
import torch


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, idx):
    y_real = copy.deepcopy(y_true)
    correct = (y_pred == y_real).sum().item()
    acc = correct / len(y_pred)

    y_pred = y_pred.squeeze()
    # Convert multi-classification to binary classification.
    y_real[y_real >= 1] = 1
    y_pred[y_pred >= 1] = 1
    correct = (y_pred == y_real).sum().item()
    binary_acc = correct / len(y_pred)

    return {'acc': acc, 'binary_acc': binary_acc, 'auc': -1, 'eer': -1, 'ap': -1,
            'pred': y_pred, 'video_auc': -1, 'label': y_real}
