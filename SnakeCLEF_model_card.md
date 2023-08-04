# Model Card

## Model Description

**Input:** Each input to the model consists of an image of a snake, a country code giving the location of the observation, and a boolean indicating whether the snake is endemic to its region.

**Output:** The algorithm outputs probabilities for the snake belonging to each of 1784 different snake species.

**Model Architecture:** The model uses a convolutional neural network, with three steps of convolution followed by max pooling and three fully connected layers. The final max pool layer is a PyTorch adaptive max pooling layer designed for dealing with images of variable size. The output of the adaptive max pooling is combined with the non-image data to form the input to the first fully connected layer. The weights are learnt through applying stochastic gradient descent to a cross-entropy loss function.

The algorithm uses three hyperparameters: the learning rate and momentum of the SGD and the weight given to the venomous snakes in the loss function. The hyperparameters are optimised using eight low-fidelity runs each using 5% of the training data; the most successful model is then trained on 50% of the training data to produce the final algorithm.

## Performance

A typical performance for the model is predicting about 6.5% of snakes correctly on the validation set, with the SnakeCLEF loss function divided by the amount of training data ("score") around 1.65. By comparison, random guessing tends to predict less than 0.1% of snakes correctly and give a score around 1.77. The algorithm also outperforms guessing the most common species for all instances. Occasionally, as a result of the stochastic training process, performance can be significantly worse. 

## Limitations

The model is only for demonstration purposes and its performance is not sufficient for it to have any practical use.

## Trade-offs

I have not studied the performance of the algorithm on individual instances and so do not know what kinds of inputs are more likely to be predicted successfully.
