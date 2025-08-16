# %%
import numpy as np

# Load the .npz file
data = np.load("dist_GAN_Real.npz")

# List all arrays stored in the file
print(data.files)

print(data)
# %%
a = data['dist']

print(len(a))
print(a)
# %%
