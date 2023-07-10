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
import pandas as pd
from io import BytesIO


# Return a random label between 0 - 9
def getRandomLabel(X_test):
    index = random.randint(0, len(X_test) - 1)
    return X_test[index]


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
    np_array = np.load(BytesIO(data), allow_pickle=True)

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
st.header("Demonstration")
index = random.randint(0, len(X_test) - 1)  # Index 0 to 9999
st.write(
    f"With X_test length of 10000 data we pick image at location {index + 1}: "
)  # Index 0 is 1 etc. as to not confused people
fig_demo, ax_demo = plt.subplots()
fig_demo.suptitle(f"Label: {Y_test[index]}")
ax_demo.imshow(X_test[index], cmap="gray")
st.pyplot(fig_demo)
st.write(
    """
         We will call the above image **handwritten_digit** where handwritten_digit is a 28x28 array of pixels values (0 - 255).
         """
)

# Loading the model
model_url = "https://github.com/Purinat33/CNN-Classroom/raw/master/mnist_cnn.h5"
model_name = "mnist_cnn.h5"
model = load_model(model_url, model_name)


st.subheader("Predicting")
st.write("Passing the image to the model is simply done using: ")
st.code("predictions = model.predict(handwritten_digit)", language="python")
st.write(
    "Which will give out 10 values, each of which representing the *probability* of a digit the handwritten_digit array represents."
)
st.write(
    "But firstly we need to reshape the input image into a shape the model expects.\n"
)
st.write(f"Current Input Shape: {X_test[index].shape}")
st.write("\nExpected Shape: (1, 28, 28, 1)")

st.subheader("Getting The Probabilities of each labels")

demo_input = X_test[index].reshape((1, 28, 28, 1))
demo_predictions = model.predict(demo_input)
predict_result = []
labels = [i for i, pred in enumerate(demo_predictions[0])]
for i, pred in enumerate(demo_predictions[0]):
    # st.write(f"Label {i}: {pred:.4f}") # Show probability of prediction being belong to what class (0-9)
    predict_result.append(round(pred * 100, 3))

results = pd.DataFrame(predict_result, index=labels, columns=["Percentage"])
results.index.name = "Label"
st.write(results)
st.write(
    f"According to the probabilities table above, we can see that the model belives the handwritten image above most likely represents the digit {results['Percentage'].idxmax()} with {results['Percentage'].max()}% probability."
)

# Get all the minimum values
# since idxmax() and idxmin() only returns the first occurence, we instead use a filter
min_percentage = results["Percentage"].min()
min_percentage_indexes = results[results["Percentage"] == min_percentage].index
st.write(
    f"Conversely, the model believes that the digit the image **least likely** to represent is/are {list(min_percentage_indexes.values)} with {results['Percentage'].min()}% of it/them being the represented digit."
)

st.divider()
st.header("Model Performance Evaluation")
st.write(
    """
    Information taken from:
[fiddler.ai](https://www.fiddler.ai/model-evaluation-in-model-monitoring/what-is-model-performance-evaluation#:~:text=In%20machine%20learning%2C%20model%20performance,metrics%20like%20classification%20and%20regression)
         """
)
st.write(
    """
    ### Loss:
    Loss is the penalty for a bad prediction. That is, loss is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. [Source](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss#:~:text=Loss%20is%20the%20penalty%20for,otherwise%2C%20the%20loss%20is%20greater)
"""
)

# Plot loss graph
numpy_url = (
    "https://github.com/Purinat33/CNN-Classroom/raw/master/mnist_cnn_history.npy"
)
