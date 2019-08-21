import os
import numpy as np
import torch.utils.data as data
import torch

MOVE_DATA_DIR = '/mnt/sda2/kennard/datasets/cg4002_dataset/old_data'
# MOVE_DATA_DIR = '/mnt/sdb2/kennard/datasets/cg4002_datasets/old_data'
train_data_dir = os.path.join(MOVE_DATA_DIR, 'train')
test_data_dir = os.path.join(MOVE_DATA_DIR, 'test')
MOVE_LIST = ['chicken', 'number7', 'wipers']

ACCEL_RANGE = 4. * 9.81  #TODO: change this once the code has been debugged.
GYRO_RANGE = 131.  #TODO: change this once the code has been debugged.
NUM_SENSOR_READINGS = 12


def read_data(is_train):
    """ Reads data """
    data_dir = train_data_dir if is_train else test_data_dir
    filenames = sorted(os.listdir(data_dir))
    filepaths = [os.path.join(data_dir, filename) for filename in filenames]
    sensor_readings, labels = list(), list()
    print(filepaths)
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
    def __init__(self, readings, labels, normalize, segment_length, num_coefficients=8):
        super(MovesDataset, self).__init__()
        self.labels = labels
        self.normalize = normalize
        self.segment_length = segment_length
        self.num_coefficients = num_coefficients

        if self.normalize:
            for sequence_readings in readings:
                sequence_readings[:,:3] /= ACCEL_RANGE
                sequence_readings[:,3:6] /= GYRO_RANGE
                sequence_readings[:,6:9] /= ACCEL_RANGE
                sequence_readings[:,9:] /= GYRO_RANGE
        self.readings = readings

        self.x, self.y = self._get_data()

    def _get_data(self):
        x, y = list(), list()
        for i, sequence_readings in enumerate(self.readings):
            label = self.labels[i]
            sequence_x = list()
            for j in range(len(sequence_readings) - self.segment_length):
                sequence_x.append(sequence_readings[j:j + self.segment_length])
            sequence_y = np.array([1] * len(sequence_x)) * label
            x += sequence_x
            y += sequence_y.tolist()

        x, y = np.array(x), np.concatenate(y, axis=0).reshape(-1)
        fourier_transformed_x = np.fft.fft(x, n=self.num_coefficients // 2, axis=1)
        fourier_transformed_x = fourier_transformed_x.reshape(-1, self.num_coefficients // 2 * NUM_SENSOR_READINGS)

        x = list()
        for transformed_x in fourier_transformed_x:
            real_values = [transformed_coefficients.real for transformed_coefficients in transformed_x]
            imag_values = [transformed_coefficients.imag for transformed_coefficients in transformed_x]
            x.append(real_values + imag_values)
        x = np.array(x)

        # make data uniform distribution -> equal distribution of each dance move
        sorted_x = [list() for _ in range(len(np.unique(self.labels)))]
        for i, label in enumerate(np.unique(self.labels)):
            has_label = y == label
            sorted_x[i] = x[has_label]

        num_data_per_class = [len(data_x) for data_x in sorted_x]
        min_data = min(num_data_per_class)

        selected_idxs = [np.random.choice(np.arange(len(data_x)), min_data) for data_x in sorted_x]
        selected_data = [data_x[selected_idxs[i]] for i, data_x in enumerate(sorted_x)]
        selected_labels = [np.array([1] * min_data) * label for label in np.unique(self.labels)]
        x = np.array(selected_data).reshape(-1, self.num_coefficients * NUM_SENSOR_READINGS)
        y = np.array(selected_labels).reshape(-1, 1)
        return x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.Tensor(self.y[idx])

    @staticmethod
    def collate_fn(batch):
        features, labels = zip(*batch)
        features = data.dataloader.default_collate(features)
        labels = data.dataloader.default_collate(labels)
        return features, labels


def get_dataset(is_train, normalize=True, segment_length=32, num_coefficients=8):
    readings, labels = read_data(is_train=is_train)
    dataset = MovesDataset(readings, labels, normalize, segment_length, num_coefficients)
    return dataset


if __name__ == '__main__':
    dataset = get_dataset(is_train=True)
    dataloader = data.DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=dataset.collate_fn)
    for (features, labels) in dataloader:
        print(features.shape)
        print(labels.shape)
