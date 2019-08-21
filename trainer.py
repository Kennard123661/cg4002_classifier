import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import LinearModel

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


class Trainer:
    def __init__(self, checkpoint_name, num_classes, num_input_features, max_epochs=100, lr=1e-2, weight_decay=5e-2):
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, self.checkpoint_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.max_epochs = max_epochs
        self.epoch = 0
        self.lr = lr
        self.weight_decay = weight_decay

        self.num_classes = num_classes
        self.num_input_features = num_input_features
        self.model = LinearModel(num_classes=self.num_classes, num_input_features=self.num_input_features)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True, weight_decay=weight_decay)

    def train(self, train_dataset, test_dataset, batch_size, eval_batch_size):
        train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_data_loader = data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False)
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.max_epochs):
            self.epoch += 1
            print("===== Epoch {} =====".format(self.epoch))
            self.train_step(train_data_loader)
            self.evaluate_step(test_data_loader)

    def train_step(self, train_data_loader):
        epoch_losses = list()
        for (features, labels) in train_data_loader:
            # print(features.shape)
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.loss_fn(predictions, labels.long().view(-1))
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
        print("loss: {}".format(np.mean(epoch_losses)))

    def evaluate_step(self, test_data_loader):
        num_correct = 0
        num_data = 0
        with torch.no_grad():
            for (features, labels) in test_data_loader:
                predictons = self.model(features)
                predictions = torch.argmax(predictons, dim=1)  # take the argmax over the class dimension N x C

                is_correct = torch.eq(predictions.view(-1), labels.long().view(-1)).int()
                num_correct += torch.sum(is_correct).item()
                num_data += labels.size(0)
        accuracy = num_correct / num_data
        print('accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    from moves_data import get_dataset, NUM_SENSOR_READINGS, MOVE_LIST
    seg_length = 32
    num_coefficients = 16

    train_dataset = get_dataset(is_train=True, segment_length=seg_length, num_coefficients=num_coefficients)
    test_dataset = get_dataset(is_train=False, segment_length=seg_length, num_coefficients=num_coefficients)

    num_input_features = num_coefficients * NUM_SENSOR_READINGS
    num_classes = len(MOVE_LIST)
    checkpoint = 'draft'

    trainer = Trainer(checkpoint_name=checkpoint, num_input_features=num_input_features, num_classes=num_classes)
    trainer.train(train_dataset, train_dataset, batch_size=500, eval_batch_size=1000)
