import json

data = json.load(open('../../../datasets/iMaterialistFashion/anns/train.json','r'))
labels = {}
total = 0
for c, image in enumerate(data["annotations"]):
    gt_labels = image["labelId"]
    for l in gt_labels:
        total += 1
        if l not in labels:
            labels[l] = 1
        else:
            labels[l] += 1

for k,v in labels.iteritems():
    labels[k] =  (float(total) / float(v))

maximum = max(labels.values())
mean = sum(labels.values()) / len(labels.values())
for k,v in labels.iteritems():
    labels[k] = v / maximum

print labels
print(max(labels.values()))
print(min(labels.values()))

with open('label_distribution.json','w') as outfile:
    json.dump(labels, outfile)
print "DONE"