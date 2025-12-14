import numpy as np
import random
import shutil
import torch

shutil.rmtree(".test/", ignore_errors=True)

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"