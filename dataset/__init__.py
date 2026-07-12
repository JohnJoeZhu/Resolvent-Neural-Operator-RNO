from .rno_dataset import *
from .grid_dataset import *
# from .prepare_data import prepare_data

DATASET_DICT={
    "car": CarCFD,
    'pendulum': Grid1DDataset,
    'lorenz': Grid1DDataset,
    'duffing': Grid1DDataset,
    'beam': Grid2DDataset,
    'diffusion': Grid2DDataset,
    'reacdiffusion': Grid2DDataset,
    'turbulent': WellDataset,
}