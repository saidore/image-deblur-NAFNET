# Image Deblurring with Modified NAFNet
This project implements a lightweight image deblurring system based on a modified NAFNet (Nonlinear Activation-Free Network). It restores sharp images from motion-blurred inputs using a fast and efficient deep learning model trained on the GoPro image deblurring dataset.

The goal was to find the best balance between image quality and compute cost, so the model can run even on limited GPUs.

This work is based on the NAFNet architecture and improves it through careful tuning of the model structure, loss functions, and training strategy

## What this project does

The system takes a blurred image and outputs a sharper version using a neural network trained on real blur examples.

It uses:

* A U-Net style NAFNet backbone

* Residual learning (the model predicts the difference between blurred and sharp)

* A custom loss function that mixes:

  * Charbonnier loss (stable pixel accuracy)

  * L1 loss (pixel difference)

  * SSIM (structure preservation)

  * Edge loss (sharp boundaries)

These are combined into one loss that encourages both sharpness and visual quality.

## What was changed from standard NAFNet

Compared to the original NAFNet, this project adds:

* GELU activations for smoother feature learning

* Configurable encoder, middle, and decoder blocks

* A multi-loss training objective (Charbonnier + SSIM + Edge + L1)

* Cosine-annealing learning rate for faster convergence

These changes improved PSNR while keeping training fast and memory-efficient.

## Dataset

The model is trained on the GoPro Image Deblurring Dataset:

* 3,214 image pairs

* Resolution: 1280 × 720

* 2,103 training images

* 1,111 test images

Each sample has a blurred image and its sharp ground truth.

## How to run (Google Colab)

1. Open Google Colab

2. Upload nafnet_blur_reduction_final.py (or open the notebook version)

3. Runtime → Change runtime type → set GPU

4. Run all cells (Runtime → Run all)

What happens when you run it:
* Installs dependencies (numpy, pytorch-msssim, torchvision, etc.)
  
* Clones the CONTRIQUE repo for feature extraction 

* Downloads the GoPro Large dataset via kagglehub 

* Extracts CONTRIQUE features and saves image_features.csv 

* Trains NAFNet on a subset of GoPro and evaluates on test pairs

Outputs you should see/save:
* image_features.csv (feature extraction output) 

* nafnet_results.png (visual results grid) 

* nafnet_deblur_model.pth (trained model weights)

## Best model configuration

After testing many settings, the best results came from:

* Width = 32

* Encoder blocks = [1, 1, 2, 6]

* Loss weights

* Charbonnier α = 1

* SSIM β = 0.2

* Edge γ = 0.01

* L1 weight = 0.05

* Learning rate = Cosine annealing

This achieved about 24.4 dB PSNR in ~18 minutes of training on limited hardware.

# What is in this repo

* nafnet_blur_reduction_final.py

  Full PyTorch implementation of the modified NAFNet, dataset loader, loss functions, and training loop 


* Paper & slides

  Explain the theory, experiments, and results
