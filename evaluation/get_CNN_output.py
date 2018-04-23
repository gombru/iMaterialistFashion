import os
import caffe
import numpy as np
from PIL import Image


# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

split = 'val'
model_name = 'iMaterialistFashion_Inception_iter_200000'

#Output file
output_file_path = '../../../ssd2/iMaterialistFashion/CNN_output/' + model_name + '/' + split + '.txt'
output_file = open(output_file_path, "w")

# load net
net = caffe.Net('deploy.prototxt', '../../../ssd2/iMaterialistFashion/models/CNN/' + model_name + '.caffemodel', caffe.TEST)


print 'Computing  ...'

count = 0

for f in os.listdir('../../../ssd2/iMaterialistFashion/img_' + split):

    count = count + 1
    if count % 100 == 0:
        print count

    # load image
    filename = '../../../ssd2/iMaterialistFashion/img_'+split+'/' + f
    im = Image.open(filename)
    im_o = im
    im = im.resize((227, 227), Image.ANTIALIAS)

    # Turn grayscale images to 3 channels
    if (im.size.__len__() == 2):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)

    #switch to BGR and substract mean
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((103.939, 116.779, 123.68))
    in_ = in_.transpose((2,0,1))


    net.blobs['data'].data[...] = in_

    # run net and take scores
    net.forward()

    # Compute SoftMax HeatMap
    topic_probs = net.blobs['output'].data[0]   #Text score

    topic_probs_str = ''

    for t in topic_probs:
        topic_probs_str = topic_probs_str + ',' + str(t)

    output_file.write(f + topic_probs_str + '\n')

output_file.close()



