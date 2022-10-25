"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     :
"""


from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from src.paths import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pprint
from src.modelling.nn_utilities import get_metrics_dict2, mean_metrics2, get_print_metrics, print_train_test_metrics, EmbeddingDataset


class NeuralNet(nn.Module):
    def __init__(self, model, experiment, output_layer=1):

        super(NeuralNet, self).__init__()

        self.experiment = experiment

        self.input_layer = experiment["input_layer"]
        self.hidden_layer = self.input_layer/2
        self.output_layer = output_layer
        self.device = experiment["device"]
        self.model_name = experiment["model_name"]
        self.model = model

        self.data = experiment.get("data", None)
        self.labels = experiment.get("labels", None)

        self.learning_rate = experiment["learning_rate"]
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        print(self.optimizer, self.learning_rate)
        self.loss_fn = nn.BCELoss()
        self.start_epoch = 0
        self.max_epochs = experiment["max_epochs"]
        self.best_loss_test = 1e10
        self.best_epoch_no = 0

        if experiment["hot_start_file_name"] is not None:
            self.load_checkpoint(experiment["hot_start_file_name"])

        self.model.to(self.device)

        self.model_directory = os.path.join(MODEL_DIRECTORY, self.experiment["experiment"], self.experiment["model_name"])
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)

        self.log_file = None


    def run_train_test(self, train_data_loader, test_data_loader):
        for i in range(self.start_epoch, self.max_epochs):
            # TRAINING

            train_metrics = []

            for batch_id, (data, label) in enumerate(train_data_loader):
                train_batch_metrics = self.train_step(data.to(self.device), label.to(self.device))
                train_metrics.append(train_batch_metrics)

            train_metrics = mean_metrics2(train_metrics)


            # Testing

            test_metrics = []

            for batch_id, (data, label) in enumerate(test_data_loader):
                test_batch_metrics = self.test_step(data.to(self.device), label.to(self.device))
                test_metrics.append(test_batch_metrics)

            test_metrics = mean_metrics2(test_metrics)

            print(get_print_metrics(test_metrics))

            if self.experiment["log_metrics"]:
                self.log_run_metrics(i, train_metrics, test_metrics)

            if self.experiment["save_checkpoints"]:
                self.check_save_checkpoint(i, test_metrics["loss"])

            self.get_best_metrics(i, test_metrics["loss"])

            if self.best_epoch_no == i:
                self.best_metrics = test_metrics


        return self.best_metrics


    def run_test(self, test_data_loader):
        test_metrics = []

        for batch_id, (data, label) in enumerate(test_data_loader):
            test_batch_metrics = self.test_step(data.to(self.device), label.to(self.device))
            test_metrics.append(test_batch_metrics)

        test_metrics = mean_metrics2(test_metrics)

        print(get_print_metrics(test_metrics))

        return test_metrics



    def train_step(self, features, labels):
        self.model.train()

        predictions = self.model(features).view(-1)

        labels = labels.float().view(-1).to(self.device)

        loss_train = self.loss_fn(predictions, labels)

        loss_train.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        train_metrics = get_metrics_dict2(predictions.detach().cpu().numpy(), labels.cpu().numpy())
        train_metrics["loss"] = loss_train.item()

        return train_metrics

    def test_step(self, features, labels):
        with torch.no_grad():
            self.model.eval()

            predictions = self.model(features).view(-1)

            labels = labels.float().view(-1).to(self.device)

            loss_test = self.loss_fn(predictions, labels)

            test_metrics = get_metrics_dict2(predictions.detach().cpu().numpy(), labels.cpu().numpy())
            test_metrics["loss"] = loss_test.item()

        return test_metrics


    def check_create_log_file(self):
        if self.log_file is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file_dir = os.path.join(LOG_DIRECTORY, self.experiment["experiment"])
            if not os.path.exists(log_file_dir):
                os.makedirs(log_file_dir)
            log_file_name = os.path.join(log_file_dir, self.model_name + "_" + timestamp + ".log")
            self.log_file = open(log_file_name, "w", encoding="utf-8")

            self.log_file.write(pprint.pformat(self.experiment, indent=4, sort_dicts=True) + "\n")

            model_stats = summary(self.model, verbose=1, depth=5, col_names=["num_params"])
            summary_str = str(model_stats)
            self.log_file.write(summary_str + "\n")

            print(f"Logging progress to: {log_file_name}")
            text = " ".join([
                "Epoch".ljust(6),
                "Timestamp".ljust(22),
                "Tr Loss".ljust(10),
                "Tr AUC".ljust(10),
                "Tr Acc".ljust(10),
                "Tr Pre".ljust(10),
                "Tr Rec".ljust(10),
                "Tr F1".ljust(10),
                "Tr JI".ljust(10),
                "Tr MCC".ljust(10),
                "Te Loss".ljust(10),
                "Te AUC".ljust(10),
                "Te Acc".ljust(10),
                "Te Pre".ljust(10),
                "Te Rec".ljust(10),
                "Te F1".ljust(10),
                "Te JI".ljust(10),
                "Te MCC".ljust(10),
                "Best".ljust(2),
                "\n"
            ])
            self.log_file.write(text)

    def log_run_metrics(self, epoch, train_metrics, test_metrics):
        self.check_create_log_file()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if test_metrics is not None:
            is_best = 1 if (test_metrics["loss"] < self.best_loss_test) else 0
            text = " ".join([
                str(epoch).ljust(6),
                timestamp.ljust(22),
                f"{train_metrics['loss']:6.4f}".ljust(10),
                f"{train_metrics['AUC']:6.4f}".ljust(10),
                f"{train_metrics['Accuracy']:6.4f}".ljust(10),
                f"{train_metrics['Precision']:6.4f}".ljust(10),
                f"{train_metrics['Recall']:6.4f}".ljust(10),
                f"{train_metrics['F1']:6.4f}".ljust(10),
                f"{train_metrics['JI']:6.4f}".ljust(10),
                f"{train_metrics['MCC']:6.4f}".ljust(10),
                f"{test_metrics['loss']:6.4f}".ljust(10),
                f"{test_metrics['AUC']:6.4f}".ljust(10),
                f"{test_metrics['Accuracy']:6.4f}".ljust(10),
                f"{test_metrics['Precision']:6.4f}".ljust(10),
                f"{test_metrics['Recall']:6.4f}".ljust(10),
                f"{test_metrics['F1']:6.4f}".ljust(10),
                f"{test_metrics['JI']:6.4f}".ljust(10),
                f"{test_metrics['MCC']:6.4f}".ljust(10),
                f"{is_best:6.4f}".ljust(2),
                "\n"
            ])
            self.log_file.write(text)
        else:
            is_best = 1 if (train_metrics["loss"] < self.best_loss_test) else 0
            text = " ".join([
                str(epoch).ljust(6),
                timestamp.ljust(22),
                f"{train_metrics['loss']:6.4f}".ljust(10),
                f"{train_metrics['AUC']:6.4f}".ljust(10),
                f"{train_metrics['Accuracy']:6.4f}".ljust(10),
                f"{train_metrics['Precision']:6.4f}".ljust(10),
                f"{train_metrics['Recall']:6.4f}".ljust(10),
                f"{train_metrics['F1']:6.4f}".ljust(10),
                f"{train_metrics['JI']:6.4f}".ljust(10),
                f"{train_metrics['MCC']:6.4f}".ljust(10),
                "\n"
            ])
            self.log_file.write(text)

        self.log_file.flush()

        print_train_test_metrics(epoch, train_metrics, test_metrics, is_best)


    def check_save_checkpoint(self, epoch_no, current_loss):
        if current_loss < self.best_loss_test:
            self.best_epoch_no = epoch_no
            self.best_loss_test = current_loss
            file_name = os.path.join(self.model_directory, f"{self.model_name}_{epoch_no}.pt.tar")
            self.save_checkpoint(epoch_no, current_loss, file_name)


    def get_best_metrics(self, epoch_no, current_loss):
        if current_loss < self.best_loss_test:
            self.best_epoch_no = epoch_no
            self.best_loss_test = current_loss


    def save_checkpoint(self, epoch_no, current_loss, path):
        checkpoint = {
            'epoch_no': epoch_no,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': current_loss,
            'best_loss': self.best_loss_test
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        self.model.to(self.device)

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.experiment["learning_rate"], amsgrad=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch_no"] + 1
        self.best_loss_test = checkpoint["best_loss"]


    def main_train_test(self):
        train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=0.2, random_state=0)
        train_dataset = EmbeddingDataset(train_data, train_labels, self.experiment["device"])
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.experiment["batch_size"], shuffle=self.experiment["shuffle_data"])

        test_dataset = EmbeddingDataset(test_data, test_labels, self.experiment["device"])
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.experiment["batch_size"], shuffle=self.experiment["shuffle_data"])

        trainer = NeuralNet(self.model, self.experiment)

        best_metrics = trainer.run_train_test(train_loader, test_loader)

        return best_metrics


    def main_test(self):
        test_dataset = EmbeddingDataset(self.data, self.labels, self.experiment["device"])
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.experiment["batch_size"], shuffle=self.experiment["shuffle_data"])

        trainer = NeuralNet(self.model, self.experiment)

        trainer.run_test(test_loader)


    def main_test_on_data(self, test_data, test_labels):
        test_dataset = EmbeddingDataset(test_data, test_labels, self.experiment["device"])
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.experiment["batch_size"], shuffle=self.experiment["shuffle_data"])

        trainer = NeuralNet(self.model, self.experiment)

        test_metrics = trainer.run_test(test_loader)

        return test_metrics




    def main_train_test_on_data(self, train_data, test_data, train_labels, test_labels):
        # train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=0.2, random_state=0)
        train_dataset = EmbeddingDataset(train_data, train_labels, self.experiment["device"])
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.experiment["batch_size"], shuffle=self.experiment["shuffle_data"])

        test_dataset = EmbeddingDataset(test_data, test_labels, self.experiment["device"])
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.experiment["batch_size"], shuffle=self.experiment["shuffle_data"])

        trainer = NeuralNet(self.model, self.experiment)

        best_metrics = trainer.run_train_test(train_loader, test_loader)

        return best_metrics