# 🧠 Fashion Item Generation using GAN

A Deep Learning project that uses a **Generative Adversarial Network (GAN)** to generate realistic fashion item images such as shirts, shoes, bags, and dresses using the Fashion-MNIST dataset.

This project demonstrates how GANs work by training two neural networks:
- 🎨 Generator
- 🕵️ Discriminator

Both networks compete with each other to generate realistic clothing images.

---

# 🚀 Features

- Generate fashion item images using GAN
- Uses Fashion-MNIST dataset
- Fast training with small dataset subset
- Saves generated images after every epoch
- Beginner-friendly implementation
- GPU support using PyTorch
- Clean project structure

---

# 🧾 Dataset

Dataset Used:
- Fashion-MNIST Dataset

Fashion-MNIST contains:
- T-shirts
- Shoes
- Bags
- Dresses
- Trousers
- Shirts

Image Details:
- 28x28 grayscale images
- Small dataset size
- Perfect for fast GAN training

Dataset is automatically downloaded using torchvision.

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

---

# 📁 Project Structure

```bash
fashion_gan_project/
│
├── data/
│   └── fashion-mnist/
│
├── models/
│   ├── generator.py
│   └── discriminator.py
│
├── outputs/
│   └── generated_images/
│
├── config.py
├── utils.py
├── train.py
└── README.md
```

---

# 🧠 GAN Architecture

## 🎨 Generator
The Generator creates fake fashion images from random noise.

### Layers:
- Linear Layer
- ReLU Activation
- Tanh Output Layer

### Input:
- Random noise vector (100 dimensions)

### Output:
- 28x28 generated fashion image

---

## 🕵️ Discriminator
The Discriminator checks whether an image is real or fake.

### Layers:
- Flatten Layer
- Linear Layers
- LeakyReLU Activation
- Sigmoid Output

### Output:
- Real or Fake probability

---

# 🔄 Training Process

The GAN training happens in two steps:

## Step 1: Train Discriminator
- Real images → Label = 1
- Fake images → Label = 0

Discriminator learns to detect fake images.

---

## Step 2: Train Generator
Generator creates fake images and tries to fool the Discriminator.

Generator improves image quality over time.

---

# ⚡ Fast Training Optimization

To reduce training time, only a small subset of the dataset is used.

```python
small_dataset = Subset(dataset, range(5000))
```

Benefits:
- Faster training
- Lower memory usage
- Works well on CPU

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/fashion-gan-project.git
cd fashion-gan-project
```

---

## Install Dependencies

```bash
pip install torch torchvision matplotlib
```

---

# ▶️ Run Project

```bash
python train.py
```

---

# 📊 Output

Generated images are saved inside:

```bash
outputs/generated_images/
```

Example:
- epoch_1.png
- epoch_5.png
- epoch_10.png

The generated images improve after each epoch.

---

# 📈 Learning Outcomes

This project helps understand:

- Deep Learning fundamentals
- GAN architecture
- Generator vs Discriminator
- Image generation
- Adversarial training
- PyTorch implementation

---

# 🎯 Future Improvements

- DCGAN implementation
- Conditional GAN
- Higher resolution images
- Streamlit web application
- Real-time image generation

---

# 👨‍💻 Author

Developed as a Deep Learning project using GAN and PyTorch.

---

# ⭐ Conclusion

This project demonstrates how Generative Adversarial Networks can learn image distributions and generate realistic fashion item images from random noise.

It is a great beginner-friendly project for learning:
- GANs
- Deep Learning
- Image Generation
- PyTorch

---
