# Infer the best theshold per class using validation set

import numpy as np
import json

split = 'val'
model_name = '1st_50k_2nd_65k_8crops'
thresholds = [0, 0.01, 0.05, 0.1, 0.12, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
              0.28, 0.29, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

num_iterations = 10

with open('../../../ssd2/iMaterialistFashion/anns/' + split + '.json', 'r') as f:
    gt_data = json.load(f)
CNN_out = open('../../../ssd2/iMaterialistFashion/CNN_output/' + model_name + '/' + split + '.txt', 'r')
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

# Iterate over all classes X times
for i in range(0,num_iterations):
    # Optimize threshold for each class
    for c in classes:
        F = -1
        # Compute F for each of the given thresholds
        for threshold in thresholds:
            f_mean_images = 0
            # Compute mean f over all images
            for el in CNN_out_data:
                tp = 0
                cur_positives = []
                el_gt = gt_data['annotations']
                data = el.split(',')
                id = int(data[0][:-4])
                values = np.zeros(228)
                for i in range(0, 228):
                    values[i] = data[i + 1]

                # Get positives
                for i, v in enumerate(values):
                    if v > thresholds_results[i]: cur_positives.append(i + 1)

                # Get f score for this image
                for pos in cur_positives:
                    if pos in gt[id]: tp+=1

                num_gt_labels = len(gt[id])
                fp = len(cur_positives) - tp
                fn = num_gt_labels - tp

                if cur_positives == 0: p = 0
                else: p = tp / cur_positives

                r = tp / num_gt_labels

                if p + r == 0:
                    f = 0
                else:
                    f = 2 * (p * r) / (p + r)

                f_mean_images += f

            f_mean_images /= len(CNN_out_data)

            if f_mean_images > F:
                F = f_mean_images
                thresholds_results[c] = threshold

        print "Iteration " + str(i) + ", Class " + str(c) + " --> " + str(thresholds_results[c]) + " Global F: " + str(F)

    json.dump(thresholds_results, open('tresholds_per_class_per_image_newopt.json','w'))