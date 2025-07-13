


# DCGAN on MNIST - Improved Training Version

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using TensorFlow and Keras to generate realistic-looking handwritten digits, trained on the MNIST dataset. The model is optimized for better performance on Kaggle and includes GIF generation of output images over training epochs.

## 📌 Features

- ✅ DCGAN built with TensorFlow 2.x
- ✅ Improved generator and discriminator architectures
- ✅ Normalized data to range [-1, 1] for better GAN training
- ✅ Image saving at each epoch for visual tracking
- ✅ Automatic checkpointing every 10 epochs
- ✅ Final output: GIF showing generator progress
- ✅ Adjustable batch size and epochs

## 📁 Project Structure

```

├── dcgan\_mnist\_final.py       # Main training script (your provided code)
├── image\_at\_epoch\_\*.png       # Generated images saved after each epoch
├── dcgan.gif                  # Animation of generated images over epochs
├── training\_checkpoints/      # Saved generator/discriminator weights
├── README.md                  # Project documentation

````

## 📦 Requirements

Make sure you have the following libraries installed:

```bash
pip install tensorflow matplotlib imageio pillow
````

## ▶️ How to Run

1. **Clone the repository** (if this is hosted on GitHub):

   ```bash
   git clone https://github.com/yourusername/dcgan-mnist-final.git
   cd dcgan-mnist-final
   ```

2. **Run the training script**:

   ```bash
   python dcgan_mnist_final.py
   ```

3. **Output**:

   * Image grid is saved as `image_at_epoch_XXXX.png` after each epoch.
   * A final GIF of generated images is saved as `dcgan.gif`.

## 🧠 Model Architecture

### Generator

* Dense → BatchNorm → LeakyReLU
* Reshape to 7×7×256
* Conv2DTranspose (128) → BN → LeakyReLU
* Conv2DTranspose (64) → BN → LeakyReLU
* Conv2DTranspose (1, tanh)

### Discriminator

* Conv2D (64) → LeakyReLU → Dropout
* Conv2D (128) → LeakyReLU → Dropout
* Flatten → Dense (1)

## 🧪 Training Settings

* **Batch Size**: 128
* **Epochs**: 50
* **Noise Vector Dim**: 100
* **Optimizer**: Adam (lr=1e-4, beta\_1=0.5)

## 🖼 Sample Results

You can view the final generated images and the progression of training in the `dcgan.gif` animation.

## 💾 Checkpoints

Training checkpoints are saved every 10 epochs in the `training_checkpoints` folder and can be used to resume or fine-tune training.

## 📊 Dataset

* MNIST Handwritten Digits
* Automatically downloaded via TensorFlow's dataset API
* Shape: `(28, 28, 1)`
* Preprocessing: Rescaled to \[-1, 1]

## 🛠 Future Improvements

* Add learning rate decay
* Try different normalization techniques
* Use Wasserstein loss for more stable training
* Extend to color image generation (e.g., CIFAR-10)

## 📝 License

This project is open-source and free to use under the MIT License.



```

---

Let me know if you want a **PDF version**, or if you'd like this turned into a GitHub repository with additional files like `requirements.txt`.
```
