from utils.geometry.operator import get_operators
from utils.geometry.curvature import get_curvs
from utils.geometry.graph import get_edges
from utils.geometry.distance import get_dists
if __name__ == "__main__": 
    get_operators('data/car/test.h5',128, compute_size=True)
    get_operators('data/car/val.h5',128, compute_size=True)
    get_operators('data/car/train.h5',128,compute_size=True)
    # get_dists('data/car/resample_test.h5',128)
    # get_edges('data/human/test.h5')
    # get_dists('data/poissonunstruc/processed/plain_unstruc.h5')