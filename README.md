# 🧠 CIFAR-10 Image Generation using GAN (PyTorch)

A **Generative Adversarial Network (GAN)** built from scratch using **PyTorch** to generate realistic-looking images from the **CIFAR-10 dataset**.
This project demonstrates the fundamental concept of adversarial learning — where a *Generator* learns to create fake images, while a *Discriminator* learns to distinguish between real and fake samples.


## 🚀 Project Overview

In this project, a GAN is trained on the **CIFAR-10 dataset** (60,000 images across 10 object classes) to generate synthetic 32×32 RGB images that resemble natural scenes and objects.

The project involves:

* Loading and normalizing the CIFAR-10 dataset
* Building the **Generator** and **Discriminator** networks
* Implementing the **adversarial training loop**
* Visualizing generated samples after each epoch


## 🧩 Architecture

### **Generator**

* Input: Random noise vector (latent dimension = 100)
* Layers: Linear → BatchNorm → ReLU → Upsampling → Conv2D → Tanh
* Output: 3×32×32 image
* Goal: Generate realistic images to fool the discriminator

### **Discriminator**

* Input: 3×32×32 image (real or generated)
* Layers: Conv2D → LeakyReLU → Dropout → Linear → Sigmoid
* Output: Probability (real or fake)
* Goal: Correctly classify real vs fake images



## 📊 Dataset

* **Dataset:** CIFAR-10
* **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
* **Training Samples:** 50,000
* **Test Samples:** 10,000
* **Image Size:** 32×32×3
* **Normalization:** [-1, 1] for compatibility with Tanh activation


## ⚙️ Requirements

Make sure you have the following dependencies installed:

```bash
pip install torch torchvision matplotlib numpy
```



## 🧠 Training the Model

1. **Clone this repository**

   ```bash
   git clone https://github.com/<your-username>/cifar10-gan-pytorch.git
   cd cifar10-gan-pytorch
   ```

2. **Run the training script**

   ```bash
   python train_gan.py
   ```

3. **During training**, generated image grids are saved after each epoch in:

   ```
   ./gan_samples/
   ```

4. **Adjust hyperparameters** in the script:

   ```python
   latent_dim = 100
   lr = 0.0002
   beta1 = 0.5
   beta2 = 0.999
   num_epochs = 50
   batch_size = 64
   ```

---

## 🖼️ Results

After training for a few epochs, the generator begins to create images resembling CIFAR-10 objects such as animals and vehicles.

| Epoch | Generated Samples                 |
| ----- | --------------------------------- |
| 1     | ![epoch1](samples/epoch_001.png)  |
| 5     | ![epoch5](samples/epoch_005.png)  |
| 10    | ![epoch10](samples/epoch_010.png) |

> *Note: Image quality improves significantly after extended training (≥100 epochs) and architectural tuning.*



## 📈 Training Logs (Example)

```
Epoch [1/10] Batch 100/1563 D Loss: 0.52 G Loss: 1.30
Epoch [5/10] Batch 800/1563 D Loss: 0.54 G Loss: 1.14
Epoch [10/10] Batch 1500/1563 D Loss: 0.45 G Loss: 1.25
```

The discriminator and generator losses fluctuate — an indicator of healthy adversarial training.



## 🎯 Key Learnings

* GANs can **learn complex visual data distributions** from scratch.
* **Normalization, weight initialization, and optimizer selection** are critical for stable training.
* **Visual monitoring** of generated images helps detect mode collapse and convergence patterns.


