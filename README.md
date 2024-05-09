# Convolutional Neural Network for Image Localization

This repository contains code for training a Convolutional Neural Network (CNN) to localize points within a 50x50 grid. The CNN is trained on a dataset of 1000 samples, where each sample consists of a 50x50 image with a single point marked.

## Prerequisites

To run the code in this repository, you need the following:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib


## Install using pip :

### pip install tensorflow numpy matplotlib

## Code Structure

- `generate_sample()`: This function generates a random (x, y) coordinate within the 50x50 grid and creates a corresponding image with a single pixel set to 255 at that coordinate.

- `create_model()`: This function defines the architecture of the CNN model using TensorFlow's Keras API. The model consists of convolutional and max-pooling layers followed by dense layers for regression.

- `train_images` and `val_images`: These variables store the training and validation images, respectively.

- `train_labels` and `val_labels`: These variables store the corresponding (x, y) coordinates for the training and validation images.

- `model.compile()`: This function compiles the model with the Adam optimizer and Mean Squared Error (MSE) loss function.

- `model.fit()`: This function trains the model on the training data for 10 epochs with a batch size of 32.

- `plt.imshow()`: This function visualizes the ground truth (GT) and predicted points on a subset of validation images.
