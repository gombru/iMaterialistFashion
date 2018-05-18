import caffe
import numpy as np

class CustomSoftmaxLoss(caffe.Layer):
    """
    Compute Cross Entropy loss with Softmax activations for multi-label classifications, accepting real numbers as labels
    Inspired by https://arxiv.org/abs/1805.00932
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance (inference and labels).")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Infered scores and labels must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)


    def forward(self, bottom, top):
        labels = bottom[1].data
        scores = bottom[0].data

        # Compute Softmax activations
        #normalizing to avoid instability
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # Store softmax activations

        logprobs = np.zeros([bottom[0].num,1])

        # Compute cross-entropy loss
        for r in range(bottom[0].num): # For each element in the batch
            scale_factor = 1 / float(np.count_nonzero(labels[r, :]))
            print("Scale facotor: " + str(scale_factor))
            print("labels[0,0]: " + str(labels[0,0]))
            print("labels[6,5]: " + str(labels[6,5]))

            for c in range(len(labels[r,:])): # For each class we compute the cross-entropy loss using the Softmax activation
                if labels[r,c] != 0:
                    logprobs[r] += -np.log(probs[r,c]) * labels[r,c] * scale_factor # We sum the loss per class for each element of the batch

        data_loss = np.sum(logprobs) / bottom[0].num

        self.diff[...] = probs
        top[0].data[...] = data_loss


    def backward(self, top, propagate_down, bottom):
        delta = self.diff   # If the class label is 0, the gradient is equal to probs due to the scale_factor weighting
        scores = bottom[0].data
        labels = bottom[1].data

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                for r in range(bottom[0].num):  # For each element in the batch
                    scale_factor = 1 / float(np.count_nonzero(labels[r, :]))
                    for c in range(len(labels[r,:])):  # For each class
                        if labels[r, c] != 0:  # If the class label != 0
                            delta[r, c] = scale_factor * (delta[r, c] - 1) + (1 - scale_factor) * delta[r, c] # Gradient for classes with positive labels considering scale factor

                bottom[i].diff[...] = delta / bottom[0].num