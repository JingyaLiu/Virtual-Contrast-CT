import numpy as np
import glob
from scipy.misc import imread


# path to images
path  = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/preprocess/data/l*/' +'*.png'
lines = glob.glob(path)
vals = np.empty(shape = (262144,))
for line in lines:
    print(line)
    image = imread(line)
    val = np.reshape(image, -1)
    print(val.max(),val.min())
    vals = np.add(vals,val)
    print(vals.shape)
print('mean, std',np.mean(vals)/len(lines), np.std(vals)/len(lines))
