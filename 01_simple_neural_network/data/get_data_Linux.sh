#!/bin/bash

# Remove previous file versions
rm -f mnist_train.csv
rm -f mnist_test.csv

# Download MNIST training data
wget https://pjreddie.com/media/files/mnist_train.csv

# Download MNIST test data
wget https://pjreddie.com/media/files/mnist_test.csv