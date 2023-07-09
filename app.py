# Convolutional Neural Network
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.datasets import mnist
import urllib.request
import keras.utils
import random


# We load data this way to `cache` the data
@st.cache_data
def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    return (X_train, Y_train), (X_test, Y_test)


# Loading the image
@st.cache_data
def load_img(url, filename):
    urllib.request.urlretrieve(url, filename)
    image = Image.open(filename)
    return image


# Loading the numpy history data
@st.cache_data
def load_np_history(url):
    # Fetch the file from the GitHub repository
    response = urllib.request.urlopen(url)
    data = response.read()

    # Convert the fetched data to a NumPy array
    np_array = np.frombuffer(data, dtype=np.float32)  # Adjust dtype if necessary

    return np_array


# Loading the model
@st.cache_resource
def load_model(url, filename):
    file_path = keras.utils.get_file(filename, origin=url)
    model = keras.models.load_model(file_path)
    return model


st.title("Convolutional Neural Network (CNN)")
st.write(
    "This application will allow users to learn what a CNN is, while having the ability to use their own input as well!"
)

st.divider()

st.header("Introduction")
st.write(
    """
![Overview of a CNN](https://editor.analyticsvidhya.com/uploads/59954intro%20to%20CNN.JPG)
"""
)
st.subheader("What is a Convolutional Neural Network?")
st.write(
    """
Convolutional Neural Network is a specialized neural network designed for visual data, such as images & videos.

For more information, please visit: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2022/01/convolutional-neural-network-an-overview/) (Image/Article)
         """
)
st.divider()

# MNIST data
(X_train, Y_train), (X_test, Y_test) = load_mnist()
st.subheader("What visual data are being used for this app?")
fig_intro, ax_intro = plt.subplots()
ax_intro.imshow(X_train[69], cmap="gray")
ax_intro.set_xlabel(f"Label: {Y_train[69]}")
st.pyplot(fig_intro)
st.write(
    """
**Answer:** the MNIST Handwritten Digits *dataset* ([Read More](https://en.wikipedia.org/wiki/MNIST_database#:~:text=The%20MNIST%20database%20(Modified%20National,the%20field%20of%20machine%20learning.)))
"""
)
st.divider()

st.write("X Data (Handwritten Images): ")
st.write(X_train[1])
st.write(
    "Each block represents a pixel of the 28x28 pixels image. With 0 being absolute black and 255 being absolute white."
)
st.write("With Y data representing the corresponding image's label. (0 to 9)")

st.divider()
st.subheader("Training and Testing")
st.write(
    "With 60000 images of 28x28 pixels for model training and another 10000 for validation, we defined the model as follow (From top to bottom):"
)

img_url = "https://github.com/Purinat33/CNN-Classroom/raw/master/CNN_mnist.png"
img_name = "CNN_mnist.png"
img = load_img(img_url, img_name)
st.image(img)
st.write(
    "The input being one of the 60000 images (or 10000 during testing) and the output being the probability of the image representing each label."
)
st.subheader("Layers Overview:")
st.write(
    """
         Each layer:
         * Conv2D: 2D convolution layer (e.g. spatial convolution over images). ([Source](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D))
         * MaxPooling2D: Downsamples the input along its spatial dimensions ([Source](https://keras.io/api/layers/pooling_layers/max_pooling2d/))
         * Flatten: Flattens the input. Does not affect the batch size. ([Source](https://keras.io/api/layers/reshaping_layers/flatten/))
         * Dense: Just your regular densely-connected NN layer. ([Source](https://keras.io/api/layers/core_layers/dense/))
         """
)

st.write(
    "From the model diagram, the model is able to extract important features and reducing the space complexity at the same time into a simple, yet informative arrays."
)
st.write(
    "Then we feed it to the Denses section for Classification into probabilities of it being 1 of the 10 classes (labels)."
)

st.divider()
st.header("Try It Yourself!")
st.write(
    "Enough talking, you can try selecting one of the test data and see how the model performed."
)
st.write("This is not yet the section where you can upload your own drawing :<")
