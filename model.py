import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, num_classes, num_input_features, hidden_len=8):
        super(LinearModel, self).__init__()
        self.num_classes = num_classes
        self.num_input_features = num_input_features

        layers = [
            nn.Linear(self.num_input_features, hidden_len),
            # nn.BatchNorm1d(hidden_len),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_len, hidden_len),
            nn.Dropout(0.5),
            # nn.BatchNorm1d(hidden_len),
            nn.ReLU(inplace=True),
        ]

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.)

        prediction_layer = nn.Linear(hidden_len, self.num_classes)
        nn.init.xavier_normal_(prediction_layer.weight)
        nn.init.constant_(prediction_layer.bias, 0.)
        layers.append(prediction_layer)
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        return self.network(input)
