# Infer the best theshold per class using validation set

import numpy as np
import json

split = 'val'
model_name = '2nd_10k_1sk_50k_8crops'
thresholds = [0, 0.01, 0.05, 0.1, 0.12, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
              0.28, 0.29, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
    F = -1
    for threshold in thresholds:
        f_mean_images = 0
        for el in CNN_out_data:
            positive = False
            true = False
            el_gt = gt_data['annotations']
            data = el.split(',')
            id = int(data[0][:-4])
            value_curClass = data[c + 1]

            if float(value_curClass) > float(threshold):
                positive = True

            # Get GT
            if c + 1 in gt[id]:
                true = True

            # Get num elements
            num_gt_labels = len(gt[id])

            # As we are computing F score per class, there will be at least this fn
            fn = num_gt_labels - 1

            if positive and true: tp = 1
            if positive and not true: fp = 1
            if not positive and true: fn += 1

            # Compute image F-Score for that threshold
            # print "tp: " + str(tp) + " fp: " + str(fp) + " fn:" + str(fn)
            if tp + fp == 0:
                p = 1

            else:
                p = tp / (tp + fp)

            if tp + fn == 0:
                r = 1

            else:
                r = tp / (tp + fn)

            if p + r == 0:
                f = 0
            else:
                f = 2 * (p * r) / (p + r)

            f_mean_images += f

        if f_mean_images > F:
            F = f_mean_images
            results[c] = threshold
            # print "TH " + str(threshold) + " --> " + "F: " + str(f)

    print "Best threshold for " + str(c) + " --> " + str(results[c]) + " F: " + str(F)

json.dump(results, 'tresholds_per_class.json')