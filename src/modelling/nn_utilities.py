"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     :
"""


from sklearn import metrics
import numpy as np
from torch.utils.data import Dataset
import torch
from datetime import datetime


def mean_metrics2(metrics):
    mean = {
        "loss": np.mean([metric["loss"] for metric in metrics]),
        "AUC": np.mean([metric["AUC"] for metric in metrics]),
        "Accuracy": np.mean([metric["Accuracy"] for metric in metrics]),
        "Precision": np.mean([metric["Precision"] for metric in metrics]),
        "Recall": np.mean([metric["Recall"] for metric in metrics]),
        "F1": np.mean([metric["F1"] for metric in metrics]),
        "JI": np.mean([metric["JI"] for metric in metrics]),
        "MCC": np.mean([metric["MCC"] for metric in metrics])
    }
    return mean


def get_metrics_dict2(probabilities, labels):
    try:
        roc_auc = metrics.roc_auc_score(labels, probabilities)
        predictions = np.rint(probabilities)
        accuracy = metrics.accuracy_score(labels, predictions)
        precision = metrics.precision_score(labels, predictions, zero_division=0)
        recall = metrics.recall_score(labels, predictions)
        f1 = metrics.f1_score(labels, predictions)
        ji = metrics.jaccard_score(labels, predictions)
        mcc = metrics.matthews_corrcoef(labels, predictions)
    except:
        roc_auc = 0
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        ji = 0
        mcc = 0
        valid_sample = 0

    metrics_dict = {
        "AUC": roc_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "JI": ji,
        "MCC": mcc
    }

    return metrics_dict

def get_print_metrics(metrics):
    metrics_string = f"Loss: {metrics['loss']:.5f} " \
                     f"AUC: {metrics['AUC']:.5f} " \
                     f"Acc: {metrics['Accuracy']:.5f} " \
                     f"Pre: {metrics['Precision']:.5f} " \
                     f"Rec: {metrics['Recall']:.5f} " \
                     f"F1: {metrics['F1']:.5f} " \
                     f"JI: {metrics['JI']:.5f} " \
                     f"MCC: {metrics['MCC']:.5f} "

    return metrics_string


def print_train_test_metrics(epoch, train_metrics, test_metrics, is_best):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if test_metrics is not None:
        print(f"{epoch} {timestamp} Train "
              f"{get_print_metrics(train_metrics)}" 
              f"Test "
              f"{get_print_metrics(test_metrics)}"
              f"Best: {is_best}")
    else:
        print(f"{epoch} {timestamp} Train "
              f"{get_print_metrics(train_metrics)}")


class EmbeddingDataset(Dataset):
    def __init__(self, data, labels, device):
        self.data = torch.from_numpy(data.astype(np.float32)).float()
        self.labels = torch.from_numpy(labels).float()
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.len