import numpy as np

x = np.random.random([10, 4])
print(x[:, 3].shape)
for i in x:
    print(i)