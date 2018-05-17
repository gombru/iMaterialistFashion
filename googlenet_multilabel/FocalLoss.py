import caffe
import numpy as np

class FocalLoss(caffe.Layer):
    """
    Compute Focal Loss
    Inspired by https://arxiv.org/abs/1708.02002
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance (inference and labels).")

        # Get Focusing Parameter
        # Adjusts the rate at which easy samples are down-weighted. WHen is 0, Focal Loss is equivalent to Cross-Entorpy.
        # Range is [0-5] 2 Leads to optimum performance in original paper
        params = eval(self.param_str)
        self.focusing_parameter = int(params['focusing_parameter'])

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Infered scores and labels must have the same dimension.")
        self.focusing_factors = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)


    def forward(self, bottom, top):
        labels = bottom[1].data
        scores = bottom[0].data

        logprobs = np.zeros([bottom[0].num,1])

        # Compute cross-entropy loss
        for r in range(bottom[0].num): # For each element in the batch
            for c in range(len(labels[r,:])): # For each class we compute the cross-entropy loss
                if labels[r,c] != 0:
                    focusing_factor = (1 - scores[r,c]) ** self.focusing_parameter
                    self.focusing_factors[r,c] = focusing_factor
                    logprobs[r] += -np.log(scores[r,c]) * labels[r,c] * focusing_factor  # We sum the loss per class for each element of the batch
                    # The class balancing factor can be included in labels by using scaled real values instead of binary labels.

        data_loss = np.sum(logprobs) / bottom[0].num

        top[0].data[...] = data_loss


    def backward(self, top, propagate_down, bottom):
        delta = np.zeros_like(bottom[0].data, dtype=np.float32)  # If the class label is 0, the gradient is 0
        scores = bottom[0].data
        labels = bottom[1].data

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                for r in range(bottom[0].num):  # For each element in the batch
                    for c in range(len(labels[r,:])):  # For each class
                        if labels[r, c] != 0:  # If the class label != 0
                            p = scores[r,c]
                            g = self.focusing_factors[r,c]
                            delta[r, c] =  labels[r,c] * ((1 - p) ** g * (g * p * np.log(p) + p - 1))# Gradient for classes with positive labels

                bottom[i].diff[...] = delta / bottom[0].num