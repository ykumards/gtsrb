# Load model from models/<whatever>

import matplotlib.pyplot as plt
import csv
import torchfile
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage import io

def imResize(img):
    return resize(img, (48, 48))

images = []

testData = torchfile.load('data/test.t7')

# Refer for index values (lua arrays are 1-indexed, so this is 1 less)
# classId, track, file = r[8], r[0], r[1]
# For labels
#dataset[idx][8]
# Ref legend:
# 0: track; 1: fileName; 2: Width; 3: Height;
# 4: X1; 5: Y1; 6: X2; 7: Y2; 8: label
prefix = 'data/test_images/'
for row in testData:
    fileName = row[0]
    x1, y1, x2, y2 = row[3], row[4], row[5], row[6]
    pathName = prefix + '%05d.ppm' % (fileName)
    img = io.imread(pathName)
    # Crop image on bounding box
    img_crop = img[x1:x2+2, y1:y2+2]
    images.append(imResize(img_crop))

X_test = np.asarray(images)

np.save('data/X_test_48', X_test)

