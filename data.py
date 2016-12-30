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
labels = []
trainData = torchfile.load('data/train.t7')
testData = torchfile.load('data/test.t7')

# Refer for index values (lua arrays are 1-indexed, so this is 1 less)
# classId, track, file = r[8], r[0], r[1]
# For labels
#dataset[idx][8]
# Ref legend:
# 0: track; 1: fileName; 2: Width; 3: Height;
# 4: X1; 5: Y1; 6: X2; 7: Y2; 8: label
prefix = 'data/train_images/'
for row in trainData:
    classId, track, fileName = row[8], row[0], row[1]
    x1, y1, x2, y2 = row[4], row[5], row[6], row[7]
    pathName = prefix + '%05d/%05d_%05d.ppm' % (classId, track, fileName)
    img = io.imread(pathName)
    # Crop image on bounding box
    img_crop = img[x1:x2+2, y1:y2+2]
    images.append(imResize(img_crop))
    labels.append(classId)

labels_a = np.asarray(images)

# Splitting the validation data (90/10)
tr_size = int(len(images_a)*0.9)
indices = np.random.permutation(images_a.shape[0])
training_idx, test_idx = indices[:tr_size], indices[tr_size:]
X_train, X_valid = images_a[training_idx,:], images_a[test_idx,:]
Y_train, Y_valid = [labels[i] for i in training_idx], [labels[i] for i in test_idx]
Y_train = np.asarray(Y_train)
Y_valid = np.asarray(Y_valid)

# Get numpy arrays

np.save('data/X_train_48', X_train)
np.save('data/Y_train', Y_train)
np.save('data/X_valid_48', X_valid)
np.save('data/Y_valid', Y_valid)

