# Simplified GAN on MNIST

This repository contains the implementation of a simplified Generative Adversarial Network (GAN) trained on the MNIST dataset. The project aims to provide an easy-to-understand introduction to GANs by using simplified architectures and hyperparameters.

## Project Overview

Generative Adversarial Networks (GANs) are a class of machine learning frameworks where two neural networks, the generator and the discriminator, are trained simultaneously through adversarial processes. The generator creates fake images to fool the discriminator, while the discriminator learns to distinguish between real and fake images.

## Project Structure

- **`gan_mnist.py`**: Main script containing the implementation and training loop for the GAN.
- **`README.md`**: Project documentation (this file).

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

You can install the required packages using pip:

```sh
pip install torch torchvision matplotlib numpy
```

## Running the Project

To train the GAN, simply run the `gan_mnist.py` script:

```sh
python gan_mnist.py
```

The script will:
1. Load and preprocess the MNIST dataset.
2. Define and initialize the generator and discriminator networks.
3. Train the GAN for a specified number of epochs.
4. Visualize the generator and discriminator losses.
5. Save generated images at each epoch.

## Hyperparameters

The script uses the following simplified hyperparameters:

- Latent Dimension: 64
- Hidden Dimension: 256
- Image Size: 28 (native MNIST size)
- Batch Size: 32
- Learning Rate: 0.0002
- Number of Epochs: 10

## Network Architectures

### Generator

The generator network takes a latent vector as input and produces a 28x28 grayscale image. It uses fully connected layers and ReLU activations.

### Discriminator

The discriminator network takes a 28x28 grayscale image as input and outputs a probability of whether the image is real or fake. It uses fully connected layers and ReLU activations.

## Visualization

The script includes functions for visualizing the training progress:

- **Loss Plot**: Shows the generator and discriminator losses over time.
- **Generated Images**: Displays generated images at each epoch.

## Results

Training the GAN on the MNIST dataset will result in the generation of handwritten digit images. The generated images will be saved as `gan_generated_images.png`, and the loss plot will be saved as `gan_loss_plot.png`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
