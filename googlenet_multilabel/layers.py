import caffe

import numpy as np
from PIL import Image
from PIL import ImageOps
import time
import sys

sys.path.append('/usr/src/opencv-3.0.0-compiled/lib/')
import cv2
import random
import json


class customDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.dir = params['dir']
        self.train = params['train']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params['batch_size']
        self.resize = params['resize']
        self.resize_w = params['resize_w']
        self.resize_h = params['resize_h']
        self.crop_w = params['crop_w']
        self.crop_h = params['crop_h']
        self.crop_margin = params['crop_margin']
        self.mirror = params['mirror']
        self.rotate_prob = params['rotate_prob']
        self.rotate_angle = params['rotate_angle']
        self.HSV_prob = params['HSV_prob']
        self.HSV_jitter = params['HSV_jitter']
        self.color_casting_prob = params['color_casting_prob']
        self.color_casting_jitter = params['color_casting_jitter']
        self.scaling_prob = params['scaling_prob']
        self.scaling_factor = params['scaling_factor']

        self.num_classes = params['num_classes']

        print "Initialiting data layer"

        # two tops: data and label
        if len(top) != 3:
            raise Exception(
                "Need to define three tops: data, multi-classification label and single label (to print acc).")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        with open(self.dir + self.split + '.json', 'r') as f:
            data = json.load(f)

        num_elements = len(data["annotations"])
        print "Number of images: " + str(num_elements)

        # Read class balances
        with open('../dataset_analysis/label_distribution.json', 'r') as f:
            class_balances = json.load(f)
        print("WARNING: BALANCING CLASSES")

        # Load labels for multiclass
        self.indices = np.empty([num_elements], dtype="S50")
        self.labels = np.zeros((num_elements, self.num_classes), dtype=np.float32)
        self.labels_single = np.zeros((num_elements, 1), dtype=np.int16)

        for c, image in enumerate(data["annotations"]):

            gt_labels = image["labelId"]
            self.indices[c] = image["imageId"]

            for l in gt_labels:
                self.labels[c, int(l) - 1] = 1 * class_balances[str(l)] * 10

            self.labels_single[c] = int(gt_labels[0]) - 1  # THIS MAY NOT WORK

            if c % 100000 == 0: print "Read " + str(c) + " / " + str(num_elements)

        print "Labels read."

        # make eval deterministic
        # if 'train' not in self.split and 'trainTrump' not in self.split:
        #     self.random = False

        self.idx = np.arange(self.batch_size)
        # randomization: seed and pick
        if self.random:
            print "Randomizing image order"
            random.seed(self.seed)
            for x in range(0, self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)
        else:
            for x in range(0, self.batch_size):
                self.idx[x] = x

        # reshape tops to fit
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, 3, params['crop_w'], params['crop_h'])
        top[1].reshape(self.batch_size, self.num_classes)
        top[2].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = np.zeros((self.batch_size, 3, self.crop_w, self.crop_h))
        self.label = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
        self.label_single = np.zeros((self.batch_size, 1), dtype=np.int16)

        # start = time.time()
        for x in range(0, self.batch_size):
            try:
                self.data[x,] = self.load_image(self.indices[self.idx[x]])
                self.label[x,] = self.labels[self.idx[x],]
                self.label_single[x] = self.labels_single[self.idx[x]]
            except:
                c = 0
                while c < self.batch_size:
                    # print("Failed loading image: " + str(self.indices[self.idx[c]]))
                    try:
                        self.data[x,] = self.load_image(self.indices[self.idx[c]])
                        self.label[x,] = self.labels[self.idx[c],]
                        self.label_single[x] = self.labels_single[self.idx[c]]
                        # print("Loaded (" + str(self.split) + ")")
                        break
                    except:
                        c += 1
                        continue

                        # print "\nLabel Single Example"
                        # print self.label_single[0,]

                        # print "\nLabel Example"
                        # print self.label[0,]

                        # end = time.time()
                        # print "Time Read IMG, LABEL and dat augmentation: " + str((end-start))

    def forward(self, bottom, top):
        # assign output
        # start = time.time()

        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.label_single

        self.idx = np.arange(self.batch_size)

        # pick next input
        if self.random:
            for x in range(0, self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)

        else:
            for x in range(0, self.batch_size):
                self.idx[x] = self.idx[x] + self.batch_size

            if self.idx[self.batch_size - 1] == len(self.indices):
                for x in range(0, self.batch_size):
                    self.idx[x] = x

                    # end = time.time()
                    # print "Time fordward: " + str((end-start))

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        # print '{}/img/trump/{}.jpg'.format(self.dir, idx)
        # start = time.time()

        if self.split == '/anns/validation':
            im = Image.open('{}{}/{}{}'.format(self.dir, 'img_val', idx, '.jpg'))
        else:
            im = Image.open('{}{}/{}{}'.format(self.dir, 'img', idx, '.jpg'))

        # To resize try im = scipy.misc.imresize(im, self.im_shape)
        # .resize((self.resize_w, self.resize_h), Image.ANTIALIAS) # --> No longer suing this resizing, no if below
        # end = time.time()
        # print "Time load and resize image: " + str((end - start))

        if self.resize:
            if im.size[0] != self.resize_w or im.size[1] != self.resize_h:
                im = im.resize((self.resize_w, self.resize_h), Image.ANTIALIAS)

        # PIL as size returns wxh not channels
        # if( im.size.__len__() == 2):
        #     im_gray = im
        #     im = Image.new("RGB", im_gray.size)
        #     im.paste(im_gray)

        # start = time.time()
        # if self.train: #Data Aumentation

        if (self.scaling_prob is not 0):
            im = self.rescale_image(im)

        if (self.rotate_prob is not 0):
            im = self.rotate_image(im)

        if self.crop_h is not self.resize_h or self.crop_h is not self.resize_h:
            im = self.random_crop(im)

        if (self.mirror and random.randint(0, 1) == 1):
            im = self.mirror_image(im)

        if (self.HSV_prob is not 0):
            im = self.saturation_value_jitter_image(im)

        if (self.color_casting_prob is not 0):
            im = self.color_casting(im)

        # end = time.time()
        # print "Time data aumentation: " + str((end - start))
        in_ = np.array(im, dtype=np.float32)
        if (in_.shape.__len__() < 3):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)
            in_ = np.array(im, dtype=np.float32)

        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    # DATA AUMENTATION

    def random_crop(self, im):
        # Crops a random region of the image that will be used for training. Margin won't be included in crop.
        width, height = im.size
        margin = self.crop_margin
        left = random.randint(margin, width - self.crop_w - 1 - margin)
        top = random.randint(margin, height - self.crop_h - 1 - margin)
        im = im.crop((left, top, left + self.crop_w, top + self.crop_h))
        return im

    def mirror_image(self, im):
        return ImageOps.mirror(im)

    def rotate_image(self, im):
        if (random.random() > self.rotate_prob):
            return im
        return im.rotate(random.randint(-self.rotate_angle, self.rotate_angle))

    def saturation_value_jitter_image(self, im):
        if (random.random() > self.HSV_prob):
            return im
        # im = im.convert('HSV')
        data = np.array(im)  # "data" is a height x width x 3 numpy array
        hsv_data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
        hsv_data[:, :, 1] = hsv_data[:, :, 1] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        hsv_data[:, :, 2] = hsv_data[:, :, 2] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        data = cv2.cvtColor(hsv_data, cv2.COLOR_HSV2RGB)
        im = Image.fromarray(data, 'RGB')
        # im = im.convert('RGB')
        return im

    def rescale_image(self, im):
        if (random.random() > self.scaling_prob):
            return im
        width, height = im.size
        im = im.resize((int(width * self.scaling_factor), int(height * self.scaling_factor)), Image.ANTIALIAS)
        return im

    def color_casting(self, im):
        if (random.random() > self.color_casting_prob):
            return im
        data = np.array(im)  # "data" is a height x width x 3 numpy array
        ch = random.randint(0, 2)
        jitter = random.randint(0, self.color_casting_jitter)
        data[:, :, ch] = data[:, :, ch] + jitter
        im = Image.fromarray(data, 'RGB')
        return im