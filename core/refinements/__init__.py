from .densecrf import DenseCRF
from .pamr import PAMR

def get(name='PAMR'):
    if name == 'PAMR': # for efficient refinement
        return PAMR()