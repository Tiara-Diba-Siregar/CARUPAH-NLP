<div align="center">

# ♻️ Carupah-NLP

**Machine Learning module for Carupah: waste image classification and CarupAI, an Indonesian-language waste-management chatbot**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![NLTK](https://img.shields.io/badge/NLP-NLTK-4B8BBE)
![TensorFlow.js](https://img.shields.io/badge/Deploy-TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white)
![Bangkit](https://img.shields.io/badge/Bangkit_Academy-2023-4285F4)
![Cohort](https://img.shields.io/badge/Cohort-Machine_Learning-34A853)

</div>

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Part 1: Image Detection](#-part-1-image-detection)
- [Part 2: Chatbot (CarupAI)](#-part-2-chatbot-carupai)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Limitations and Future Work](#-limitations-and-future-work)
- [Team](#-team)
- [Acknowledgements](#-acknowledgements)

---

## 🌱 About the Project

**Carupah** is an application that helps users manage waste more easily. Users can:

- 📷 **Detect the type of waste** from a photo.
- 💬 **Chat with CarupAI**, a chatbot that answers waste-related questions.
- 📍 **Find the nearest waste bank**.

This repository contains the **two machine-learning implementations** behind the app:

| Module | Purpose | Approach |
|---|---|---|
| **Image Detection** | Classifies waste by its type | Convolutional Neural Network (CNN) |
| **Chatbot (CarupAI)** | Acts as knowledge management for waste-related questions | NLP with bag-of-words intent classification |

---

## ✨ Features

- **Waste classification** from images, based on waste type.
- **Real-time data augmentation** during training to reduce overfitting.
- **Intent-based chatbot** trained on a custom Indonesian dataset in JSON format (44 trained intents).
- **Wide waste coverage**: general waste, organic and inorganic waste, plastic (including HDPE), aluminium, glass, cardboard, styrofoam, electronic waste, and detergent waste.
- **App-ready models**: the image model is exported to **TFLite**, and the chatbot model is converted to **TensorFlow.js**.

---

## 🛠 Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | TensorFlow, Keras (v2.12) |
| NLP | NLTK (`punkt`, `wordnet`) |
| Data handling | NumPy, Split-folders, Pickle, JSON |
| Visualization | Matplotlib |
| Environment | Google Colab, Anaconda, or any Python IDE |
| Model formats | HDF5 (`.h5`), TFLite, TensorFlow.js |

---

## 📂 Repository Structure

```text
Carupah-NLP/
├── Carupah_CarupAI.ipynb     # Chatbot notebook (training, testing, TF.js conversion)
├── chatbot.json              # Intents dataset (patterns and responses)
├── chatbot_model.h5          # Trained chatbot model (Keras HDF5)
├── data_words.pkl            # Vocabulary: 107 lemmatized words
├── data_classes.pkl          # Intent classes: 44 tags
└── modeljs/                  # TensorFlow.js version of the chatbot model
    ├── model.json
    └── group1-shard1of1.bin
```

> Add your image detection files (notebook, saved models) to this tree as well.

---

## 📷 Part 1: Image Detection

The image detection feature classifies waste by type. It is built as a **Convolutional Neural Network** with **TensorFlow/Keras**, trained on a dataset of waste photos stored in the local environment.

### Workflow

**1. Set up the data**
Install and import the required packages, then locate the dataset (Google Drive or local environment). Use the **Split-folders** package to split the dataset into training and validation sets.

**2. Preprocess the data**
Define the target image height and width, then apply data augmentation with `tf.keras.preprocessing.image`:

- `ImageDataGenerator`: performs real-time augmentation while training (for example rotating, shifting, and rescaling pixel values). This helps prevent overfitting.
- `flow_from_directory`: loads images from the dataset directories, generates batches, sets the target image size, and sets the label mode. `class_mode` is set to `sparse` because the dataset has more than one class.

The training and validation datasets are then passed through the augmentation pipeline.

**3. Build the model**
The CNN is built with Keras (TensorFlow backend):

| Layer / Component | Role |
|---|---|
| `Sequential` | Builds the model by adding layers one by one in order |
| `Conv2D` | Core building block of a CNN; performs convolution on image input |
| `MaxPooling2D` | Downsamples feature maps while keeping the most important information |
| `Flatten` | Converts multi-dimensional output into a one-dimensional vector |
| `Dense` | Adds representation capacity to the model |
| `Dropout` | Reduces overfitting |
| `Activation` | Introduces the non-linearity needed to model complex data |
| **Optimizer** | RMSprop (Root Mean Square Propagation) |

**4. Train and validate**
Create the model with `create_model()`, then run `fit()` with the training and validation data.

**5. Analyze the results**
Use `matplotlib.pyplot` to plot training and validation **loss** and **accuracy** for each epoch.

**6. Save the model**
Save the trained model in **HDF5** and **TFLite** formats.

---

## 💬 Part 2: Chatbot (CarupAI)

**CarupAI** is the chatbot inside the Carupah app. It answers waste-related questions in **Bahasa Indonesia**. It is built with **Natural Language Processing** using TensorFlow, and trained on a **custom dataset in JSON format**. The notebook is designed to run in **Google Colab**.

### Dataset

| Item | Value |
|---|---|
| Intents in `chatbot.json` | 45 (44 trained, plus a `noanswer` fallback with no training patterns) |
| Training patterns | 241 |
| Responses | 89 |
| Vocabulary (lemmatized, unique) | 107 words |

Topics: greetings and small talk, general waste, organic and inorganic waste, plastic and HDPE, aluminium, glass, cardboard, styrofoam, electronic waste, and detergent waste. Each topic covers things like definitions, categories, handling, and recycled products.

### Workflow

**1. Set up the data**
Fetch the dataset from Google Drive using the `requests` library and decode the byte content into a string, then load it as JSON.

**2. Preprocess the data** (NLTK)

- **Tokenization**: splits each pattern into tokens using the `punkt` package.
- **Lemmatization**: converts words to their root form using the `wordnet` package. Words are lowercased and punctuation is removed.

The vocabulary and the class list are saved as Pickle files (`data_words.pkl`, `data_classes.pkl`).

**3. Create the training data**
Each pattern is converted into a **bag-of-words** vector (1 if the vocabulary word appears, 0 otherwise) with a one-hot label for its intent. The data is shuffled before training.

**4. Build the model**
A feed-forward network built with Keras (TensorFlow backend):

| Layer | Output Shape | Parameters |
|---|---|---|
| `Dense` (ReLU) | 256 | 27,648 |
| `Dropout` (0.5) | 256 | 0 |
| `Dense` (ReLU) | 128 | 32,896 |
| `Dropout` (0.5) | 128 | 0 |
| `Dense` (Softmax) | 44 | 5,676 |
| **Total** | | **66,220** |

- **Input:** 107-dimensional bag-of-words vector
- **Output:** probability over 44 intents
- **Loss:** categorical cross-entropy
- **Optimizer:** SGD (learning rate 0.01, decay 1e-6, momentum 0.9, Nesterov)

**5. Train the model**
`fit()` with **100 epochs** and **batch size 5**.

**6. Visualize loss and accuracy**
Plot loss and accuracy per epoch to monitor training progress.

**7. Save the model**
Save the trained model in **HDF5** format (`chatbot_model.h5`).

**8. Load the sources**
Load the dataset, the Pickle objects, and the HDF5 model.

**9. Prepare for inference**
Helper functions:

- `clean_sentence()`: tokenizes and lemmatizes the input sentence.
- `bow()`: converts a sentence into a bag-of-words vector.
- `predict_classes()`: predicts the intent. Only intents with probability above **0.25** are kept, sorted by confidence.
- `getResponse()`: picks a random response for the predicted intent.

**10. Test the chatbot**
A chat loop lets you talk to CarupAI. Type `end` to quit.

```text
CarupAI: Hai, Aku CarupAI. Aku akan membantu kamu menjawab pertanyaan tentang sampah.
You:     hai hai
CarupAI: Hai, Selamat datang. Ada yang mau kamu tanyakan?
You:     apa itu sampah
CarupAI: Sampah adalah segala jenis material atau zat yang dihasilkan oleh manusia
         atau alam yang dianggap tidak berguna, tidak diinginkan, ...
```

**11. Convert to TensorFlow.js**
Convert the model with `tensorflowjs` so it can run in the web or JavaScript environment. Output: `model.json` and `group1-shard1of1.bin`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Google Colab (recommended for the chatbot), Anaconda, or a Python IDE
- Packages: `tensorflow`, `keras`, `nltk`, `numpy`, `matplotlib`, `requests`, `split-folders`, `tensorflowjs`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install tensorflow nltk numpy matplotlib requests split-folders tensorflowjs
```

### Run the chatbot notebook

1. Open `Carupah_CarupAI.ipynb` in **Google Colab**.
2. Run the cells in order. The notebook downloads the NLTK resources automatically:
   ```python
   nltk.download('punkt')
   nltk.download('wordnet')
   ```
3. The notebook fetches the dataset from Google Drive. To use the local copy instead, load `chatbot.json` with `json.load()`.
4. Use the chat loop at the end of the notebook to test CarupAI.

### Run the image detection notebook

1. Download the waste image dataset and place it in the local environment (or Google Drive).
2. Open the image detection notebook in Anaconda or your Python IDE.
3. Run the cells in order: setup, preprocessing, model building, training, analysis, and saving.

---

## 📊 Results

| Model | Metric | Result |
|---|---|---|
| Chatbot (CarupAI) | Training accuracy (epoch 100) | **99.59 %** |
| Chatbot (CarupAI) | Training loss (epoch 100) | **0.0278** |
| Image Detection (CNN) | Validation accuracy | `XX %` *(fill in)* |

> The chatbot figures are **training** metrics on 241 patterns, not results on held-out data.

<!-- Add your plots: ![Chatbot loss and accuracy](images/chatbot-training.png) -->

---

## 🔭 Limitations and Future Work

- **Evaluation:** the chatbot is evaluated on training accuracy only. Adding a validation or test split would show how well it handles unseen questions.
- **Fallback handling:** when no intent passes the 0.25 confidence threshold, the current chat loop has no reply. The `noanswer` intent can be used as the fallback response.
- **Dataset size:** 241 patterns across 44 intents is small. More paraphrases per intent would improve robustness to different phrasing.
- **Language coverage:** the chatbot is trained on Indonesian. Slang, typos, and mixed language are only handled if present in the patterns.

---

## 👥 Team

**Carupah Team, Bangkit Academy 2023, Machine Learning Cohort**

| Name | Role | Link |
|---|---|---|
| Tiara Diba | Machine Learning | [LinkedIn](https://www.linkedin.com/in/tiara-diba-895753229/) |
| `<Team member>` | `<Role>` | `<Link>` |

---

## 🙏 Acknowledgements

- **Bangkit Academy 2023** for the learning program.
- The TensorFlow and NLTK communities for the open-source tools.
