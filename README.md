# 🧠 NVIDIA DLI — Fundamentals of Deep Learning  
### Comprehensive Notebook Collection · CNNs · Transfer Learning · NLP · Final Assessment (96.32%)  
### Completed Official NVIDIA Deep Learning Institute Course (Certificate Earned)

This repository contains all notebooks, scripts, and exercises completed during the **NVIDIA Deep Learning Institute (DLI)** training on the *Fundamentals of Deep Learning*.  

The course provides hands-on GPU-accelerated experience with:

- Image classification  
- CNN model building  
- Data augmentation  
- Transfer learning  
- Fine-tuning ImageNet models  
- Introductory NLP  
- End-to-end deep learning workflows  
- Passing a real-world assessment  

I completed the final evaluation with **96.32% validation accuracy**, surpassing the required 92%.

---

# 📌 Course Overview

The NVIDIA DLI workshop teaches modern deep learning foundations using:

- **PyTorch**
- **Torchvision**
- **GPU acceleration (CUDA)**
- **VGG16 transfer learning**
- **Custom datasets**
- **Hands-on problem solving**

The course is structured as a sequence of Jupyter notebooks, each focusing on a different DL concept.

---

# 📁 Repository Contents

All workshop notebooks included:

```
01_jupyterlab.ipynb
01_mnist.ipynb
02_asl.ipynb
03_asl_cnn.ipynb
05a_doggy_door.ipynb
05b_presidential_doggy_door.ipynb
06_nlp.ipynb
07_assessment.ipynb
utils.py
Untitled.py
untitled.md
```

Optional (recommended) media folder:

```
media/
  mnist_predictions.png
  asl_training_curves.png
  doggydoor_results.png
  fruit_classification_accuracy.png
```

---

# 📚 Key Concepts Learned

### ✔ Deep Learning Basics  
- Neurons and layers  
- Forward & backward pass  
- Loss functions  
- Backpropagation  
- SGD & Adam optimizers  

### ✔ Image Classification  
- MNIST digit classification  
- Data normalization  
- Neural network training  

### ✔ CNNs (Convolutional Neural Networks)  
- Convolution layers  
- Pooling  
- Dropout  
- Deep CNN architectures  

### ✔ Data Augmentation  
- Rotation  
- Rescaling  
- Random crop  
- Brightness & color jitter  
- Horizontal flip  

### ✔ Transfer Learning  
- Using pretrained ImageNet models  
- Freezing & unfreezing weights  
- Replacing classifier heads  
- Fine-tuning with low learning rates  

### ✔ NLP Fundamentals  
- Tokenization  
- Embeddings  
- Simple neural text classifiers  

### ✔ Final Assessment  
- Custom Dataset class  
- Full training + validation loop  
- Achieved **96.32% validation accuracy**  

---

# 📊 Notebook Summaries

## 📘 MNIST (01_mnist.ipynb)
- Built first NN classifier for handwritten digits  
- Flatten → Dense → ReLU → Softmax  
- Trained with CrossEntropyLoss  
- Achieved high accuracy  

---

## ✋ American Sign Language (02_asl.ipynb)
- Preprocessing A–Z gesture dataset  
- Built CNN for classification  
- Applied strong augmentation  
- Achieved strong accuracy  

---

## 🔥 Advanced ASL CNN (03_asl_cnn.ipynb)
- Added deeper convolution layers  
- Introduced dropout & regularization  
- Improved classification performance  

---

## 🐶 Doggy Door Classifier (05a_doggy_door.ipynb)
- Binary classification using transfer learning  
- Loaded pretrained CNN  
- Froze feature extractor  
- Replaced classifier head  
- Demonstrated feature extraction  

---

## 🐕 Presidential Doggy Door (05b_presidential_doggy_door.ipynb)
- Used pretrained **VGG16**  
- Feature extraction + fine-tuning  
- Unfroze final blocks with small LR  
- Improved overall performance  

---

## ✍️ NLP Basics (06_nlp.ipynb)
- Tokenization  
- Embedding layers  
- Feedforward text classifiers  
- Evaluating NLP models  

---

# 🏆 Final Assessment — Fruit Freshness Classification
### ✔ Required Accuracy: **92%**  
### ✔ Achieved: **96.32%**

A 6-class image classification problem:

| Class |
|-------|
| fresh apples |
| fresh bananas |
| fresh oranges |
| rotten apples |
| rotten bananas |
| rotten oranges |

Dataset structure:
```
data/fruits/train/
data/fruits/valid/
```

---

# 🛠 Model Architecture (VGG16 Transfer Learning)

### ✔ Load Pretrained VGG16
```python
from torchvision.models import vgg16, VGG16_Weights

weights = VGG16_Weights.DEFAULT
vgg_model = vgg16(weights=weights)
```

### ✔ Freeze Base Model
```python
vgg_model.requires_grad_(False)
```

### ✔ Custom Classification Head
```python
my_model = nn.Sequential(
    vgg_model.features,
    vgg_model.avgpool,
    nn.Flatten(),
    vgg_model.classifier[0:3],
    nn.Linear(4096, 500),
    nn.ReLU(),
    nn.Linear(500, 6)
)
```

### ✔ Augmentation Used
```python
RandomRotation(25)
RandomResizedCrop(224)
RandomHorizontalFlip()
ColorJitter()
```

### ✔ Fine-Tuning
```python
vgg_model.requires_grad_(True)
optimizer = Adam(my_model.parameters(), lr=1e-5)
```

### ✔ Final Evaluation
```python
utils.validate(my_model, valid_loader, valid_N, loss_function)
```

**Final Accuracy: 96.32%**

---

# 🎫 Certification

Successfully completed the **NVIDIA Deep Learning Institute — Fundamentals of Deep Learning** course.  
(Certificate not added in repository intentionally.)

---

# 📂 Project Structure

```
nvidia-dli-deeplearning/
│── 01_jupyterlab.ipynb
│── 01_mnist.ipynb
│── 02_asl.ipynb
│── 03_asl_cnn.ipynb
│── 05a_doggy_door.ipynb
│── 05b_presidential_doggy_door.ipynb
│── 06_nlp.ipynb
│── 07_assessment.ipynb
│── utils.py
│── Untitled.py
│── untitled.md
│
├── media/                
│     └── jupyterlab_environment.png
│
└── README.md
```

---

# 🚀 Skills Demonstrated

- PyTorch coding  
- Training & validation loops  
- CNN model building  
- Data preprocessing  
- Data augmentation  
- Transfer learning  
- Fine-tuning ImageNet models  
- GPU-accelerated training  
- Model evaluation (accuracy, loss, etc.)  
- Handling custom datasets  
- Applied machine learning workflow end to end  

---

# 📬 Contact

**Arnav Saxena**  
AI/ML · Deep Learning · Computer Vision  
📧 Email: **arnav12saxena@gmail.com**  
🔗 LinkedIn: https://www.linkedin.com/in/arnav-saxena-a9a217367  

---
