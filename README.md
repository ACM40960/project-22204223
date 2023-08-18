
# Atherosclerosis Detection in CT Scan Images Using Deep Learning and AI Neural Networks

## Introduction
This project aims to use deep learning and AI neural networks to detect atherosclerosis in CT scan images. Ultrasound imaging is a non-invasive, cost-effective, and widely accessible technique for visualizing the internal structures of the body. Automated detection of atherosclerosis in ultrasound images can aid in early diagnosis, risk assessment, and treatment planning.

![YdtYLY](https://github.com/ACM40960/project-22204223/assets/67566026/205e7000-fded-4cc9-a0d0-6596185de169)

<img width="677" alt= "data" src = "https://github.com/ACM40960/project-22204223/assets/67566026/6ca01715-2189-4171-bce0-8c7701ed09c1">


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

## Authors
- Abhinav Tyagi (22202296)
- Yash Vats (22204223) (ywaths@gmail.com)

## Acknowledgments

Thank you Sarp Ackay for giving the opportunity for creating this project.
