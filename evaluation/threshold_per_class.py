# Infer the best theshold per class using validation set

import numpy as np
import json

split = 'val'
model_name = '2nd_10k_1sk_50k_8crops'
thresholds = [0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25]

with open('../../../ssd2/iMaterialistFashion/anns/' + split + '.json', 'r') as f:
    gt_data = json.load(f)
CNN_out = open('../../../ssd2/iMaterialistFashion/CNN_output/' + model_name + '/' + split + '.txt', 'r')
CNN_out_data = []

# Get CNN data
for el in CNN_out:
    CNN_out_data.append(el)
gt = {}
# Get GT data
for el in gt_data['annotations']:
        id = int(el['imageId'])
        gt_labels = el['labelId']
        gt_labels_int = []
        for cur in gt_labels:
            gt_labels_int.append(int(cur))
        gt[id] = gt_labels_int

classes = np.arange(228)
results = {}

for c in classes:
    F = 0
    for threshold in thresholds:
        tp=0
        fp=0
        fn=0
        for el in CNN_out_data:
            positive = False
            true = False
            el_gt = gt_data['annotations']
            id = int(data[0][:-4])
            data = el.split(',')
            value_curClass = data[c+1]
            if value_curClass > threshold:
                positive = True

            # Get GT
            if c+1 in gt[id]: t = True

            if positive and true: tp+=1
            if positive and not true: fp+=1
            if not positive and true: fn+=1

        # Compute class F-Score for that threshold
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2 * (p * r) / (p + r)

        if f > F:
            F = f
            results[c] = threshold

    print "Best threshold for " + str(c) + " --> " + str(results[c])

json.dump(results, 'tresholds_per_class.json')
