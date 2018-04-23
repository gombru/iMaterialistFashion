import numpy as np
import json

split = 'val'
model_name = 'iMaterialistFashion_Inception_iter_200000'
threshold = 0.5

with open('../../../ssd2/iMaterialistFashion' + split + '.json', 'r') as f:
    gt_data = json.load(f)

CNN_out = open('../../../ssd2/iMaterialistFashion/CNN_output/' + model_name + '/' + split + '.txt', 'r')

results = {}

F = 0

for el in CNN_out:

    el_gt = gt_data['annotations']

    cur_idx = []
    values = np.zeros(228)
    data = el.split(',')
    id = data[0]
    for i in range(0,128):
        values[i] = int(data[i+1])

    for i,v in enumerate(values):
        if v > threshold: cur_idx += i+1

    results[id] = cur_idx

for el in gt_data['annotations']:
    tp=0
    fp=0
    id = el['imageId']
    gt_labels = el['labelId']
    out_labels = results[id]
    for l in out_labels:
        if l in gt_labels:
            tp += 1
        else:
            fp += 1
        fn = len(gt_labels) - tp
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2 * (p*r) / (p + r)
        F += f

F /= len(gt_data['annotations'])
print "F: " + str(F)
