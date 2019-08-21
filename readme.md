## Setup

Install the following dependencies:
```
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Data Format

The readings in .txt files are in the format (dx, dy, dz, roll, pitch yaw)

## Training

To train the model, simply run 
```
python trainer.py 
```