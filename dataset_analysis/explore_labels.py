import json

all_labels = {}
dataset = "../../../datasets/iMaterialistFashion/anns/validation.json"

with open(dataset, 'r') as f:
    data = json.load(f)
    for image in data["annotations"]:
        labels = image["labelId"]
        for l in labels:
            if l not in all_labels:
                all_labels[l] = 1
            else:
                all_labels[l] += 1

print all_labels
print "Total labels: " + str(len(all_labels))


