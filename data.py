import os
import numpy as np
import torch.utils.data as data

# MOVE_DATA_DIR = '/mnt/sda2/kennard/data/cg4002_dataset/old_data'
MOVE_DATA_DIR = '/mnt/sdb2/kennard/datasets/cg4002_datasets/old_data'
train_data_dir = os.path.join(MOVE_DATA_DIR, 'train')
test_data_dir = os.path.join(MOVE_DATA_DIR, 'test')
MOVE_LIST = ['chicken', 'number7', 'wipers']

ACCEL_RANGE = 4. * 9.81  #TODO: change this once the code has been debugged.
GYRO_RANGE = 131.  #TODO: change this once the code has been debugged.


def read_data(is_train):
    """ Reads data """
    data_dir = train_data_dir if is_train else test_data_dir
    filenames = sorted(os.listdir(data_dir))
    filepaths = [os.path.join(data_dir, filename) for filename in filenames]
    sensor_readings, labels = list(), list()

    for filepath in filepaths:
        filename = os.path.split(filepath)[-1]
        move_type = filename.split('_')[0]
        move_idx = np.where(np.array(MOVE_LIST) == move_type)

        with open(filepath, 'r') as f:
            sequence_readings = f.readlines()
        sequence_readings = [reading.strip().split(',') for reading in sequence_readings]
        sequence_readings = np.array(sequence_readings, dtype=np.float32)
        labels.append(move_idx)
        sensor_readings.append(sequence_readings)
    return sensor_readings, labels


class MovesDataset(data.Dataset):
    def __init__(self, readings, labels, normalize, segment_length):
        super(MovesDataset, self).__init__()
        self.labels = labels
        self.normalize = normalize
        self.segment_length = segment_length

        if self.normalize:
            for sequence_readings in readings:
                sequence_readings[:,:3] /= ACCEL_RANGE
                sequence_readings[:,3:6] /= GYRO_RANGE
                sequence_readings[:,6:9] /= ACCEL_RANGE
                sequence_readings[:,9:] /= GYRO_RANGE
        self.readings = readings

        self.x, self.y = self._get_data()
        print(np.fft.fft(self.x, n=self.segment_length, axis=1).shape)
        # print(self.x.shape)
        # print(self.y.shape)

    def _get_data(self):
        x, y = list(), list()
        for i, sequence_readings in enumerate(self.readings):
            label = self.labels[i]
            sequence_x = list()
            for j in range(len(sequence_readings) - self.segment_length):
                sequence_x.append(sequence_readings[j:j + self.segment_length])
            sequence_y = np.ones(shape=len(sequence_x)).tolist()
            x += sequence_x
            y += sequence_y
        return np.array(x), np.array(y)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def get_dataset(is_train, normalize, segment_length):
    readings, labels = read_data(is_train=is_train)
    dataset = MovesDataset(readings, labels, normalize, segment_length)
    print(dataset.readings)
    return dataset


if __name__ == '__main__':
    # readings, labels = read_data(is_train=True)
    # print(readings)
    get_dataset(True, True, segment_length=32)
