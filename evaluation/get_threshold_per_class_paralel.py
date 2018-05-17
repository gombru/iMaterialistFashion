# Infer the best theshold per class using validation set

import numpy as np
import json
from joblib import Parallel, delayed

split = 'val'
model_name = '1st_50k_2nd_65k_IN_45k_LRAD_20k_8crops'
# thresholds = [0.01, 0.05, 0.1, 0.12, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
#               0.28, 0.29, 0.3, 0.31, 0.32, 0.34, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
thresholds = []
thresholds = np.arange(0.0,0.295,0.005)
thresholds += np.arange(0.3,1,0.01)

num_iterations = 30

with open('../../../datasets/iMaterialistFashion/anns/' + 'validation' + '.json', 'r') as f:
    gt_data = json.load(f)
CNN_out = open('../../../datasets/iMaterialistFashion/' + model_name + '/' + split + '.txt', 'r')
CNN_out_data = []

# Get CNN data
for el in CNN_out: CNN_out_data.append(el)

gt = {}
# Get GT data
for el in gt_data['annotations']:
    id = int(el['imageId'])
    gt_labels = el['labelId']
    gt_labels_int = []
    for cur in gt_labels: gt_labels_int.append(int(cur))
    gt[id] = gt_labels_int

classes = np.arange(228)
np.random.shuffle(classes)

# To store output thresholds. Initialize to 0.2
thresholds_results = {}
for c in classes: thresholds_results[c] = 0.2

def compute_f_mean_class(el):
    tp = 0.0
    cur_positives = []
    data = el.split(',')
    id = int(data[0][:-4])
    values = np.zeros(228)
    for v_i in range(0, 228):
        values[v_i] = data[v_i + 1]

    # Get positives
    for i, v in enumerate(values):
        if i == c:
            if v > threshold:
                cur_positives.append(i + 1)
        elif v > thresholds_results[i]:
            cur_positives.append(i + 1)

    # Get f score for this image
    for pos in cur_positives:
        if pos in gt[id]: tp += 1

    num_gt_labels = len(gt[id])

    if len(cur_positives) == 0:
        p = 0.0
    else:
        p = tp / float(len(cur_positives))

    r = tp / float(num_gt_labels)

    if p + r == 0:
        f = 0.0
    else:
        f = 2.0 * (p * r) / (p + r)

    return f

# Iterate over all classes X times
print "Starting..."
for iter in range(0,num_iterations):
    np.random.shuffle(classes)
    # Optimize threshold for each class
    for c in classes:
        F = -1
        TH = -1
        # Compute F for each of the given thresholds
        for threshold in thresholds:
            f_mean_images = 0
            # r_mean_images = 0
            # p_mean_images = 0

            # Compute mean f over all images
            parallelizer = Parallel(n_jobs=24)
            tasks_iterator = (delayed(compute_f_mean_class)(el) for el in CNN_out_data)
            r = parallelizer(tasks_iterator)
            # merging the output of the jobs
            strings = np.vstack(r)
            for s in strings:
                f_mean_images += float(s[0])
                # p_mean_images += p
                # r_mean_images += r

            f_mean_images /= float(len(CNN_out_data))
            # p_mean_images /= float(len(CNN_out_data))
            # r_mean_images /= float(len(CNN_out_data))

            # print "TH: " + str(threshold) + " f " + str(f_mean_images) + " p " + str(p_mean_images) + " r " + str(r_mean_images)

            if f_mean_images > F:
                F = f_mean_images
                TH = threshold

        thresholds_results[c] = TH
        print "Iteration " + str(iter) + ", Class " + str(c) + " --> Threshold: " + str(thresholds_results[c]) + ", Global F: " + str(F)

    json.dump(thresholds_results, open('tresholds_per_class_it_' + str(iter) + '.json','w'))
