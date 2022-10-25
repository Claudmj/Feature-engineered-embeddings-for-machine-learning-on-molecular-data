"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     :
"""

import torch.nn as nn

class NeuralNetModel(nn.Module):
    def __init__(self, input_layer_dim=100, output_layer_dim=1):
        super(NeuralNetModel, self).__init__()

        self.input_layer_dim = input_layer_dim
        self.hidden_layer_dim = int(input_layer_dim/2)
        self.output_layer_dim = output_layer_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.input_layer_dim, self.hidden_layer_dim),
            nn.BatchNorm1d(self.hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(int(self.hidden_layer_dim), self.output_layer_dim),
            nn.Sigmoid()
        )


    def forward(self, embedding):
        predictions = self.classifier(embedding)

        return predictions





