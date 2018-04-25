# ANN-Bank
Use of ANN to predict potential bank leavers

Dataset is taken from https://www.superdatascience.com/deep-learning/ (PART 1. ARTIFICIAL NEURAL NETWORKS (ANN))
We will be using Artificial Neural Networks to predict the potential bank leavers. For that, we have dataset of around 10000 people with the attributes including their Nation, Age, Credit Score, Bank Balance, Estimated salary etcetra. So using this we will be using artificial neural network with 11 input layer dimension, 2 hidden layers(of dimensions 6 each) and output layer(with dimension 1) that returns the probability of the person to leave the bank.

We will be using Rectifier activation function for the Hidden layers and Sigmoid activation function for the output layer.

## Batch Gradient Descent
Initially without using Batch Gradient Descent we on 10 different batches of the training set (train+cross validation) to find the average accuracies over the 10 different train+cross_validation sets. It comes out to be around 84.3%.

## Regularization Dropout
In later updation, we did Regularization Dropout with a drop out probaility of 0.1. This way we are making sure that each neuron is trained in an effective way independent of the other neuron so the net accuracy of the neural net is increased. This time the accuracy was around 86%
