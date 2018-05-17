import os
import caffe
import numpy as np
from PIL import Image

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()


model = '../../../datasets/iMaterialistFashion/iMaterialistFashion_Inception_iter_100000.caffemodel'

# load net
net = caffe.Net('deploy.prototxt', model, caffe.TEST)

print 'Computing  ...'

# load image
img_name = '../../../datasets/SocialMedia/img_resized_1M/cities_instagram/london/1481255189662056249.jpg'
im = Image.open(img_name)

# Turn grayscale images to 3 channels
if (im.size.__len__() == 2):
    im_gray = im
    im = Image.new("RGB", im_gray.size)
    im.paste(im_gray)


# Crops the central sizexsize part of an image
crop = True
if crop:
    crop_size = 256
    width, height = im.size

    if width != crop_size:

        left = (width - crop_size) / 2
        right = width - left
        im = im.crop((left, 0, right, height))

    if height != crop_size:
        top = (height - crop_size) / 2
        bot = height - top
        im = im.crop((0, top, width, bot))


im = im.resize((224, 224), Image.ANTIALIAS)

# switch to BGR and substract mean
in_ = np.array(im, dtype=np.float32)
in_ = in_[:, :, ::-1]
in_ -= np.array((103.939, 116.779, 123.68))
in_ = in_.transpose((2, 0, 1))

net.blobs['data'].data[...] = in_

# run net and take scores
net.forward()

# Compute SoftMax HeatMap
topic_probs = net.blobs['output'].data[0]
print topic_probs

