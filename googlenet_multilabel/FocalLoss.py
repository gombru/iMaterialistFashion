import caffe
import numpy as np
import json

class FocalLoss(caffe.Layer):
    """
    Compute Focal Loss
    Inspired by https://arxiv.org/abs/1708.02002
    Raul Gomez Bruballa
    https://gombru.github.io/
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance (inference and labels).")

        # Get Focusing Parameter
        # Adjusts the rate at which easy samples are down-weighted. WHen is 0, Focal Loss is equivalent to Cross-Entorpy.
        # Range is [0-5] 2 Leads to optimum performance in original paper
        params = eval(self.param_str)
        self.focusing_parameter = int(params['focusing_parameter'])
        print("Focusing Paramerer: " + str(self.focusing_parameter))

        print("Reading class balances")
        with open('../dataset_analysis/label_distribution.json', 'r') as f:
            self.class_balances = json.load(f)
        print("WARNING: BALANCING CLASSES")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Infered scores and labels must have the same dimension.")
        top[0].reshape(1)

    def forward(self, bottom, top):
        labels = bottom[1].data
        scores = bottom[0].data
        # Compute sigmoid activations
        scores =  1 / (1 + np.exp(-scores))

        logprobs = np.zeros([bottom[0].num, 1])

        # Compute cross-entropy loss
        for r in range(bottom[0].num):  # For each element in the batch
            for c in range(len(labels[r, :])):
                # For each class we compute the cross-entropy loss
                # We sum the loss per class for each element of the batch
                if labels[r, c] == 0: # Loss form for negative classes
                    logprobs[r] += self.class_balances[str(c+1)] * -np.log(1-scores[r, c]) * scores[r, c] ** self.focusing_parameter
                else: # Loss form for positive classes
                    logprobs[r] += self.class_balances[str(c+1)] * -np.log(scores[r, c]) * (1 - scores[r, c]) ** self.focusing_parameter
                    # The class balancing factor can be included in labels by using scaled real values instead of binary labels.

        data_loss = np.sum(logprobs) / bottom[0].num
        top[0].data[...] = data_loss

    def backward(self, top, propagate_down, bottom):
        delta = np.zeros_like(bottom[0].data, dtype=np.float32)
        labels = bottom[1].data
        scores = bottom[0].data
        # Compute sigmoid activations
        scores =  1 / (1 + np.exp(-scores))

        for r in range(bottom[0].num):  # For each element in the batch
            for c in range(len(labels[r, :])):  # For each class
                p = scores[r, c]
                if labels[r, c] == 0:
                    delta[r, c] = self.class_balances[str(c+1)] * -(p ** self.focusing_parameter) * ((self.focusing_parameter - p * self.focusing_parameter) * np.log(1-p) - p) # Gradient for classes with negative labels
                else:  # If the class label != 0
                    delta[r, c] = self.class_balances[str(c+1)] * (((1 - p) ** self.focusing_parameter) * (
                    self.focusing_parameter * p * np.log(
                        p) + p - 1))  # Gradient for classes with positive labels

        bottom[0].diff[...] = delta / bottom[0].num