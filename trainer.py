import os

import torch
import tensorboardX
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import LinearModel

CHECKPOINT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs')


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

        # self.log_dir = os.path.join(LOG_DIR, self.checkpoint_name)
        self.log_writer = tensorboardX.SummaryWriter('logs/' + self.checkpoint_name)
        self.load_checkpoint('model.pt')

    def train(self, train_dataset, test_dataset, batch_size, eval_batch_size):
        train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_data_loader = data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False)
        start_epoch = self.epoch
        print("Training...\n")
        for epoch in range(start_epoch, self.max_epochs):
            self.epoch += 1
            self.train_step(train_data_loader)
            train_acc = self.evaluate_step(train_data_loader)
            test_acc = self.evaluate_step(test_data_loader)
            self.log_writer.add_scalars('accuracy',
                                        {
                                            'test': test_acc,
                                            'train': train_acc
                                        },
                                        self.epoch)
            if self.epoch % 20 == 0:
                print("===== Epoch {} =====".format(self.epoch))
                print('test-acc: {0}\ttrain acc: {1}'.format(test_acc, train_acc))

    def train_step(self, train_data_loader):
        self.model.train()
        epoch_losses = list()
        for (features, labels) in train_data_loader:
            # print(features.shape)
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.loss_fn(predictions, labels.long().view(-1))
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
            self.save_checkpoint('model.pt')
        self.log_writer.add_scalar('loss', np.mean(epoch_losses), self.epoch)

    def evaluate_step(self, test_data_loader):
        self.model.eval()
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
        return accuracy

    def save_checkpoint(self, filename):
        checkpoint_filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }, checkpoint_filepath)

    def load_checkpoint(self, filename):
        checkpoint_filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(checkpoint_filepath):
            checkpoint = torch.load(checkpoint_filepath)
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optim'])
            print("Loaded checkpoint")
        else:
            print("Checkpoint not found, continuing...")


if __name__ == '__main__':
    from moves_data import get_dataset, NUM_SENSOR_READINGS, MOVE_LIST
    seg_length = 32
    num_coefficients = 8

    train_dataset = get_dataset(is_train=True, segment_length=seg_length, num_coefficients=num_coefficients)
    test_dataset = get_dataset(is_train=False, segment_length=seg_length, num_coefficients=num_coefficients)

    num_input_features = num_coefficients * NUM_SENSOR_READINGS
    num_classes = len(MOVE_LIST)
    checkpoint = 'draft1'

    trainer = Trainer(checkpoint_name=checkpoint, num_input_features=num_input_features, num_classes=num_classes)
    trainer.train(train_dataset, test_dataset, batch_size=500, eval_batch_size=1000)
