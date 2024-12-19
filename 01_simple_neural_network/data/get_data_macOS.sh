#!/bin/zsh

# Remove previous file versions
rm -f mnist_train.csv
rm -f mnist_test.csv

# Download MNIST training data
curl -o mnist_train.csv https://pjreddie.com/media/files/mnist_train.csv

# Download MNIST test data
curl -o mnist_test.csv https://pjreddie.com/media/files/mnist_test.csv