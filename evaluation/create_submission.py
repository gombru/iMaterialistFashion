import numpy as np
import json

split = 'test'
model_name = 'iMaterialistFashion_Inception_iter_50000'
threshold = 0.2

out_file = open('../../../ssd2/iMaterialistFashion/submissions/' + model_name + '_' + split + '.csv', 'w')

with open('../../../ssd2/iMaterialistFashion/anns/test.json', 'r') as f:
    gt_data = json.load(f)
CNN_out = open('../../../ssd2/iMaterialistFashion/CNN_output/' + model_name + '/' + split + '.txt', 'r')

out_file.write('image_id,label_id\n')

for el in CNN_out:
    el_gt = gt_data['images']
    cur_idx = []
    values = np.zeros(228)
    data = el.split(',')
    id = data[0]
    for i in range(0, 228):
        values[i] = data[i + 1]
    line = str(int(id))
    for i, v in enumerate(values):
        if v > threshold:
            line += ',' + str(i)


