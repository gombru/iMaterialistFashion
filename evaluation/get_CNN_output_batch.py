import sys
import caffe
import numpy as np
from PIL import Image, ImageOps
import os


#switch to BGR and substract mean
def preprocess(im):
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104, 117, 123))
    in_ = in_.transpose((2,0,1))
    return in_

# Run in GPU
caffe.set_device(3)
caffe.set_mode_gpu()

split = 'val'
elements = []
for f in os.listdir('../../../ssd2/iMaterialistFashion/img_' + split):
    elements.append(f)

folder_name = '1st_50k_2nd_65k_IN_45k_LRAD_20k_8crops'

#Ensemble 2 classifiers
model = 'CNN_secondTraining/iMaterialistFashion_Inception_iter_65000'
model2 = 'CNN_firstTraining/iMaterialistFashion_Inception_iter_50000'
model3 = 'CNN_imageNet/iMaterialistFashion_Inception_iter_45000'
model4 = 'CNN_lrdecay/iMaterialistFashion_Inception_lrdecay_iter_20000'

num_crops = 8

#Output file
output_file_dir = '../../../ssd2/iMaterialistFashion/CNN_output/' + folder_name + '/'

if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)
output_file_path = output_file_dir + '/'+split+'.txt'
output_file = open(output_file_path, "w")

# load net
net = caffe.Net('deploy.prototxt', '../../../ssd2/iMaterialistFashion/models/' + model + '.caffemodel', caffe.TEST)
net2 = caffe.Net('deploy.prototxt', '../../../ssd2/iMaterialistFashion/models/' + model2 + '.caffemodel', caffe.TEST)
net3 = caffe.Net('deploy_imageNet.prototxt', '../../../ssd2/iMaterialistFashion/models/'+ model3 + '.caffemodel', caffe.TEST)
net4 = caffe.Net('deploy.prototxt', '../../../ssd2/iMaterialistFashion/models/'+ model4 + '.caffemodel', caffe.TEST)

# Reshape net
batch_size = 30
size = 224
net.blobs['data'].reshape(batch_size, 3, size, size)
net2.blobs['data'].reshape(batch_size, 3, size, size)
net3.blobs['data'].reshape(batch_size, 3, size, size)
net4.blobs['data'].reshape(batch_size, 3, size, size)


print 'Computing  ...'

count = 0
i = 0
while i < len(elements):
    indices = []

    x = 0
    # Fill batch
    while (x < batch_size - num_crops):

        if i > len(elements) - 1: break

        if i % 100 == 0:
            print i


        filename = '../../../ssd2/iMaterialistFashion/img_' + split + '/' + elements[i]
        im = Image.open(filename)
        # im_o = im

        # Turn grayscale images to 3 channels
        if (im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        width, height = im.size

        #Do crops
        crops = []

        #1 Crop 256x256 center and resize to 224x224
        crop_size = 256
        patch = im.copy()

        if width != crop_size:
            left = (width - crop_size) / 2
            right = width - left
            patch = patch.crop((left, 0, right, height))
        if height != crop_size:
            top = (height - crop_size) / 2
            bot = height - top
            patch = patch.crop((0, top, width, bot))
        crops.append(patch)

        # 2 Crop 224x224 center
        crop_size = 224
        patch = im.copy()
        left = (width - crop_size) / 2
        right = width - left
        top = (height - crop_size) / 2
        bot = height - top
        patch = patch.crop((left, top, right, bot))
        crops.append(patch)

        # 3 If wide image: Crop left and right 256x256 and resize to 224x224
        if width > height:
            patch = im.copy()
            left = 0
            right = 255
            patch = patch.crop((left, 0, right, height))
            crops.append(patch)

            patch = im.copy()
            left = width - 256
            right = width
            patch = patch.crop((left, 0, right, height))
            crops.append(patch)

        # 3 If vertical image: Crop top and bot 256x256 and resize to 224x224
        else: #height > width:
            patch = im.copy()
            top = 0
            bot = 255
            crop = patch.crop((0, top, width, bot))
            crops.append(patch)

            patch = im.copy()
            top = height - 256
            bot = height
            patch = patch.crop((0, top, width, bot))
            crops.append(patch)


        # Crops 5-8: Mirror done crops
        for c in range(0,4):
            crops.append(ImageOps.mirror(crops[c]))

        if len(crops) != num_crops: print "Warning, not " +str(num_crops)+ " crops for this image"

        for crop in crops:
            crop = crop.resize((224, 224), Image.ANTIALIAS)
            crop = preprocess(crop)
            net.blobs['data'].data[x,] = crop
            net2.blobs['data'].data[x,] = crop
            net3.blobs['data'].data[x,] = crop
            net4.blobs['data'].data[x,] = crop

            indices.append(elements[i])

            x += 1

        i += 1

    # run net and take scores
    net.forward()
    net2.forward()
    net3.forward()
    net4.forward()



    # Save results for each batch element
    x=0
    while x < len(indices):
        c=0
        probs = np.zeros(net.blobs['output'].data[0].size)
        probs2 = np.zeros(net.blobs['output'].data[0].size)
        probs3 = np.zeros(net.blobs['output'].data[0].size)
        probs4 = np.zeros(net.blobs['output'].data[0].size)


        while c < num_crops: # for each crop
            probs += net.blobs['output'].data[x+c]
            probs2 += net2.blobs['output'].data[x+c]
            probs3 += net3.blobs['output'].data[x+c]
            probs4 += net4.blobs['output'].data[x+c]

            c+=1

            # print probs.argsort()[::-1][0:5]
            # print probs2.argsort()[::-1][0:5]

        probs = probs / num_crops
        probs2 = probs2 / num_crops
        probs3 = probs3 / num_crops
        probs4 = probs4 / num_crops

        probs = (probs + probs2 + probs3 + probs4) / 4
        #probs = (probs + probs2) / 2


        #top5 = probs.argsort()[::-1][0:5]
        #top5str = ''
        string_result = ''
        for t in probs:
            string_result = string_result + ',' + str(t)

        output_file.write(indices[x] + string_result + '\n')

        x+=num_crops

output_file.close()

print "DONE"
print output_file_dir