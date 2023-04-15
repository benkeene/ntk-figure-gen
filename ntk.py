import torch
import torch.nn as nn

from functorch import make_functional, vmap, vjp, jvp, jacrev

from networks import Dense
from pytorch_code import empirical_ntk_jacobian_contraction

import IPython


