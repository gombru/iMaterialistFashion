import caffe

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()


# load net
net = caffe.Net('evaluation/deploy.prototxt', '../../datasets/iMaterialistFashion/iMaterialistFashion_Inception_iter_95000.caffemodel', caffe.TEST)

print net.blobs

