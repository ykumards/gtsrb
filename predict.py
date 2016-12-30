from __future__ import print_function

import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import torchfile

from keras.models import load_model

modelName = 'simple_aug_model.h5'

model = load_model('models/' + modelName)
X_test = np.load('data/X_test_48.npy')

testData = torchfile.load('data/test.t7')
ids = []

for row in testData:
    ids.append(row[0])

preds = model.predict_classes(X_test)

fo = csv.writer(open("outputs/simple_aug_model.csv", "w"), lineterminator="\n")
fo.writerow(["Filename","ClassId"])

for i, item in enumerate(ids):
    fo.writerow(["%05d" % (ids[i]), preds[i]])
