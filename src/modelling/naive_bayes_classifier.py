from sklearn import naive_bayes, metrics
from src.paths import *
import os
import joblib
import pickle


class nbc_master_class():
    """
    Class for NBC with methods to train, predict, evaluate, save models and load models

    Args:
        model_parameters: dict
        train_data: np.array
        test_data: np.array
        train_labels: list/np.array
        test_labels: list/np.array
        embedding_type: str
        load_model: str
    """
    def __init__(self, model_parameters, train_data, test_data, train_labels, test_labels, embedding_type, load_model=None):
        super(nbc_master_class, self).__init__()

        self.model_name = model_parameters.get("model_name", "EveAffinity")
        self.dataset_name = model_parameters.get("dataset_name", "")
        self.model_is_trained = False
        self.embedding_type = embedding_type

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

        if load_model is None:
            if self.embedding_type == "cv":
                self.model = naive_bayes.MultinomialNB()
            else:
                self.model = naive_bayes.GaussianNB()
        else:
            self.model = joblib.load(load_model)
            self.model_is_trained = True

    def train(self):
        self.model.fit(self.train_data, self.train_labels)
        self.model_is_trained = True

    def predict(self):
        if self.model_is_trained:
            self.test_predictions = self.model.predict(self.test_data)
            self.train_predictions = self.model.predict(self.train_data)

    def save_model(self, save_path):
        model_folder = os.path.join(save_path, self.model_name)
        if self.model_is_trained:
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
            with open(os.path.join(save_path, self.model_name, self.model_name + ".pkl"), 'wb') as f:
                pickle.dump(self.model, f)


    def evaluate(self):
        if self.model_is_trained:
            self.test_metrics = {
                "Accuracy": metrics.accuracy_score(self.test_labels, self.test_predictions),
                "Precision": metrics.precision_score(self.test_labels, self.test_predictions),
                "Recall": metrics.recall_score(self.test_labels, self.test_predictions),
                "F1": metrics.f1_score(self.test_labels, self.test_predictions),
                "JI": metrics.jaccard_score(self.test_labels, self.test_predictions),
                "MCC": metrics.matthews_corrcoef(self.test_labels, self.test_predictions),
                "ConfusionMatrix": metrics.confusion_matrix(self.test_labels, self.test_predictions)
            }
            try:
                self.test_metrics["AUC"] = metrics.roc_auc_score(self.test_labels, self.test_predictions)
            except:
                self.test_metrics["AUC"] = 0
            self.train_metrics = {
                "Accuracy": metrics.accuracy_score(self.train_labels, self.train_predictions),
                "Precision": metrics.precision_score(self.train_labels, self.train_predictions, zero_division=0),
                "Recall": metrics.recall_score(self.train_labels, self.train_predictions),
                "F1": metrics.f1_score(self.train_labels, self.train_predictions),
                "JI": metrics.jaccard_score(self.train_labels, self.train_predictions),
                "MCC": metrics.matthews_corrcoef(self.train_labels, self.train_predictions),
                "ConfusionMatrix": metrics.confusion_matrix(self.train_labels, self.train_predictions)
            }
            try:
                self.train_metrics["AUC"] = metrics.roc_auc_score(self.train_labels, self.train_predictions)
            except:
                self.train_metrics["AUC"] = 0

    def log_metrics(self, experiment):
        if self.test_metrics is not None:
            model_folder = os.path.join(LOG_DIRECTORY, experiment)
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
        log_file_name = os.path.join(model_folder, self.model_name + ".log")
        log_file = open(log_file_name, "w", encoding="utf-8")

        log_file.write(
            f"TRAIN:\t"
            f"Accuracy: {self.train_metrics['Accuracy']} \t"
            f"Precision: {self.train_metrics['Precision']} \t"
            f"Recall: {self.train_metrics['Recall']} \t"
            f"F1: {self.train_metrics['F1']} \t"
            f"AUC: {self.train_metrics['AUC']} \t"
            f"JI: {self.train_metrics['JI']} \t"
            f"MCC: {self.train_metrics['MCC']} \t"
            f"ConfusionMatrix: {self.train_metrics['ConfusionMatrix']} \n"

            f"TEST:\t"
            f"Accuracy: {self.test_metrics['Accuracy']} \t"
            f"Precision: {self.test_metrics['Precision']} \t"
            f"Recall: {self.test_metrics['Recall']} \t"
            f"F1: {self.test_metrics['F1']} \t"
            f"AUC: {self.test_metrics['AUC']} \t"
            f"JI: {self.test_metrics['JI']} \t"
            f"MCC: {self.test_metrics['MCC']} \t"
            f"ConfusionMatrix: {self.test_metrics['ConfusionMatrix']} \n"

            f"PARAMETERS:\t"
            f"model_name: {self.model_name} \t"
            f"dataset_name: {self.dataset_name} \n")

        log_file.flush()