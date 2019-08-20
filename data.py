import os
import numpy as np
import torch.utils.data as data

MOVE_DATA_DIR = '/mnt/sda2/kennard/data/cg4002_dataset/old_data'
MOVE_LIST = ['chicken', 'number7', 'wipers']


def read_data():
    """ Reads data """
    filenames = sorted(os.listdir(MOVE_DATA_DIR))
    filepaths = [os.path.join(MOVE_DATA_DIR, filename) for filename in filenames]
    sensor_readings, labels = list(), list()

    for filepath in filepaths:
        filename = os.path.split(filepath)[-1]
        move_type = filename.split('_')[0]
        move_idx = np.where(np.array(MOVE_LIST) == move_type)

        with open(filepath, 'r') as f:
            file_sensor_readings = f.readlines()
        file_labels = np.ones_like(file_sensor_readings, dtype=int) * move_idx
        file_sensor_readings = [sensor_reading.strip().split(',') for sensor_reading in file_sensor_readings]
        file_sensor_readings = np.array(file_sensor_readings, dtype=np.float32)
        labels += file_labels.tolist()
        sensor_readings += file_sensor_readings.tolist()
    return sensor_readings, labels


class MovesDataLoader(data.DataLoader):
    def __init__(self, data, labels):
        super(MovesDataLoader, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


if __name__ == '__main__':
    readings, labels = read_data()
    print(readings, labels)
