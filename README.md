
# Atherosclerosis Detection in CT Scan Images Using Deep Learning and AI Neural Networks


- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [UNet Architecture](#unet-architecture)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [Model Development](#model-development)
  - [Training](#training)
- [Results](#results)
  - [Accuracy and Loss Plots](#accuracy-and-loss-plots)
  - [Model Output Images](#model-output-images)
- [Conclusion and Challenges](#conclusion-and-challenges)
- [Future Use](#future-use)
- [References](#references)
- [Contributions](#contributions)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)



## Introduction
This project aims to use deep learning and AI neural networks to detect atherosclerosis in CT scan images. Ultrasound imaging is a non-invasive, cost-effective, and widely accessible technique for visualizing the internal structures of the body. Automated detection of atherosclerosis in ultrasound images can aid in early diagnosis, risk assessment, and treatment planning.

![YdtYLY](https://github.com/ACM40960/project-22204223/assets/67566026/205e7000-fded-4cc9-a0d0-6596185de169)

<img width="677" alt= "data" src = "https://github.com/ACM40960/project-22204223/assets/67566026/6ca01715-2189-4171-bce0-8c7701ed09c1" height="500">


## Project Structure
The repository contains the following files and directories:

- `images.zip`: A ZIP file containing the CT scan images used for model training and testing.
- `labels.zip`: A ZIP file containing the labels corresponding to the images in `images.zip`.
- `model.h5`: The trained model file in HDF5 format.
- `test5.ipynb`: A Jupyter Notebook file for testing the model.
- `app.py`: The Python file for the web application that predicts the probability of detecting atherosclerosis.
- `error.html`: An HTML file for error handling in the web application.
- `index.html`: The HTML file for the main page of the web application.
- `Project.pptx`: The PowerPoint presentation that provides an overview of the project.

## UNet Architecture 
UNET is a U-shaped encoder-decoder network architecture, which consists of four encoder blocks and four decoder blocks that are connected via a bridge. The encoder network (contracting path) half the spatial dimensions and double the number of filters (feature channels) at each encoder block

Here's an explanation of the U-Net architecture:
Encoder (Down sampling Path):
The encoder path extracts features from the input image through a series of convolutional layers and max-pooling operations.
At each level of the encoder, the spatial dimensions of the feature maps are reduced by half (via max-pooling), while the number of channels (feature maps) is increased.
Decoder (Up sampling Path):
The decoder path reconstructs the segmentation mask from the encoded feature maps.
At each level of the decoder, the spatial dimensions of the feature maps are increased by a factor of two (via upsampling), and the number of channels is decreased.
The decoder receives input from both the encoder and the previous layer of the decoder through skip connections.
Skip Connections:
Skip connections are used to transfer feature maps from the encoder to the corresponding level in the decoder.
These connections help preserve fine-grained spatial information that is lost during down sampling.
The feature maps from the encoder are concatenated with the feature maps in the decoder to provide additional context.
Output Layer:
The final layer of the U-Net produces a segmentation mask, where each pixel is assigned a label corresponding to a specific class.
The output layer often uses a SoftMax activation function to produce class probabilities for each pixel.

![1_jhYv-BI-dEQe85I7B4qjcQ](https://github.com/ACM40960/project-22204223/assets/67566026/9703c3bb-c5b7-41cb-9a1c-f865a2565ffd)


## Methodology

### Preprocessing
Preprocessing: The images were resized, normalized, and the labels were one-hot encoded. The dataset was split into training, validation, and test sets.
```python
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, jaccard_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
```
```python
# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images_array, labels_onehot, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

```
### Model Development
Model Development: We used the U-Net architecture, a convolutional neural network (CNN) commonly used for image segmentation tasks. Below Given code is for U-net model, you can add more layers and hyperparameters for accuracy.

<img width="677" alt="Screenshot 2023-08-17 at 5 08 43 PM" src="https://github.com/ACM40960/project-22204223/assets/67566026/62e88cc3-e43c-42a8-9c17-965dceef71bc">



### Training
Training: The model was trained on the training set using the categorical cross entropy loss function and the Adam optimizer. Also you edit the code and increase number of epochs and batch size for better results and training.

```python
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[EarlyStopping, ModelCheckpoint],
    verbose=1
)

```


## Results

### Accuracy and Loss Plots
<img width="765" alt="Screenshot 2023-08-17 at 5 16 44 PM" src="https://github.com/ACM40960/project-22204223/assets/67566026/df558513-b3fd-442b-b66f-78880257cee8">
<img width="412" alt="Screenshot 2023-08-17 at 5 26 08 PM" src="https://github.com/ACM40960/project-22204223/assets/67566026/da327c96-2f74-4e6a-b61f-f036663625d1">
<img width="334" alt="Screenshot 2023-08-17 at 7 12 11 PM" src="https://github.com/ACM40960/project-22204223/assets/67566026/4c0abac8-de37-4372-85f3-65adb0c0b3a0">

### Model Output Images

## Conclusion and Challenges

The matrix indicates that the classifier performs well in predicting Background, Plague, and Artifacts classes (high diagonal values), while having difficulty accurately predicting the Lumen classes (low diagonal values and high off-diagonal values). Challenges include high complexity (time-consuming), lack of data on the Internet, and machine hindrance.

## Future Use

We have developed a web app that predicts the probability of detecting atherosclerosis with an accuracy of 92%.

<img width="1680" alt="Screenshot 2023-08-18 at 11 48 55 PM" src="https://github.com/ACM40960/project-22204223/assets/67566026/6efe5ea3-cce9-4f8d-b03e-df8ea582b264">

<img width="1680" alt="Screenshot 2023-08-18 at 11 51 16 PM" src="https://github.com/ACM40960/project-22204223/assets/67566026/4e2e661e-82b3-46ca-ac1d-3f001e167f15">

## References

- [Performance of a Deep Neural Network Algorithm Based on a Small Medical Image Dataset](https://rdcu.be/djKW0)
- [Artificial Intelligence in Cardiovascular Atherosclerosis Imaging](https://doi.org/10.3390/jpm12030420)
- Datasets:
  - [Mayo Clinic Dataset](https://www.kaggle.com/datasets/hey24sheep/mayoclinicdataset)
  - [Atherosclerosis Segmentation](https://www.kaggle.com/datasets/davidpiln/atherosclerosis-segmentation)
    

## Contributions

1. **Fork the Project**: Click the 'Fork' button at the top right of this page to create your own copy of this repository.
2. **Clone Your Fork**: Clone your fork to your local machine using `git clone https://github.com/your-username/your-repo-name.git`.
3. **Create a New Branch**: Create a new branch for your feature or fix using `git checkout -b feature/YourFeatureName`.
4. **Make Your Changes**: Make and commit your changes with descriptive commit messages.
5. **Push to Your Fork**: Push your changes to your fork on GitHub using `git push origin feature/YourFeatureName`.
6. **Open a Pull Request**: Go to the 'Pull requests' tab on the original repository and click the 'New pull request' button. Select your fork and the branch you created, then click 'Create pull request'.
7. **Wait for Review**: We will review your pull request and provide feedback. Please be patient and address any comments or requested changes.

If you have any questions or need help, feel free to reach out to us or open an issue.

Thanks for contributing!


## Authors
- Abhinav Tyagi (22202296)
- Yash Vats (22204223) (ywaths@gmail.com)

## Acknowledgments

Thank you Sarp Ackay for giving the opportunity for creating this project.
