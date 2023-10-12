# Customize activation functions for the Siamese network.
# Description: The Siamese network outputs a difference score between two images. And the y_true labels are either 0 or 1.
# Thus, the final activation function should be a binary classifier like sigmoid or softmax.