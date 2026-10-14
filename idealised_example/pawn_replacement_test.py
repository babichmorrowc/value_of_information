# Test if I'm sampling with or without replacement
import numpy as np

N = 100
NU_unique = 10
idx = np.random.choice(np.arange(0, N, 1), size=(NU_unique, ),
                               replace='False')
idx