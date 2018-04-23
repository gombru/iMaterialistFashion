import numpy as np
import json

split = 'val'
model_name = '2nd_10k_1sk_50k_8crops'
thresholds = json.load(open('tresholds_per_class.json','r'))

with open('../../../ssd2/iMaterialistFashion/anns/' + split + '.json', 'r') as f:
    gt_data = json.load(f)
CNN_out = open('../../../ssd2/iMaterialistFashion/CNN_output/' + model_name + '/' + split + '.txt', 'r')
CNN_out_data = []
for el in CNN_out:
    CNN_out_data.append(el)


F = 0
results = {}
for el in CNN_out_data:

    el_gt = gt_data['annotations']
    cur_idx = []
    values = np.zeros(228)
    data = el.split(',')
    id = data[0]
    for i in range(0, 228):
        values[i] = data[i + 1]

    for i, v in enumerate(values):
        if v > thresholds[i]: cur_idx.append(i + 1)

    results[id[:-4]] = cur_idx

c = 0
for el in gt_data['annotations']:
    tp = 0.0
    fp = 0.0
    id = el['imageId']
    gt_labels = el['labelId']
    gt_labels_int = []
    for cur in gt_labels:
        gt_labels_int.append(int(cur))

    out_labels = results[id]
    for l in out_labels:
        if l in gt_labels_int:
            tp += 1.0
        else:
            fp += 1.0

    fn = len(gt_labels_int) - tp

    if tp + fp == 0:
        if fp == 0:
            p = 1
        else:
            p = 0
    else:
        p = tp / (tp + fp)
    if tp + fn == 0:
        if fn == 0:
            r = 1
        else:
            r = 0
    else:
        r = tp / (tp + fn)

    if p + r == 0:
        f = 0
    else:
        f = 2 * (p * r) / (p + r)

    F += f
    c += 1

F /= len(gt_data['annotations'])
print "F: " + str(F)