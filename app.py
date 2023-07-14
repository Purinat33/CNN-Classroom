# Convolutional Neural Network
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import mnist
import urllib.request
import keras.utils
import random
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import numpy as np


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


@st.cache_data
def load_history_csv(url, filename):
    urllib.request.urlretrieve(url, filename)
    df = pd.read_csv(filename)
    return df


# Loading the model
@st.cache_resource
def load_model(url, filename):
    file_path = keras.utils.get_file(filename, origin=url)
    model = keras.models.load_model(file_path)
    return model


@st.cache_data
def plot_history(history):
    st.write(
        """
        ##### Loss History:
             """
    )

    fig_loss, ax_loss = plt.subplots()
    fig_loss.suptitle("Train and Test Loss")
    ax_loss.plot(history["loss"], label="Train")
    ax_loss.plot(history["val_loss"], label="Test")
    fig_loss.supxlabel("Epoch")
    fig_loss.supylabel("Loss")
    fig_loss.legend()
    ax_loss.grid()
    st.pyplot(fig_loss)

    st.write(
        """
             ### Accuracy:
             Accuracy is the fraction of predictions our model got right. [Source](https://developers.google.com/machine-learning/crash-course/classification/accuracy)

             """
    )
    st.write(
        """
        ##### Accuracy History:
             """
    )
    # Accuracy chart
    fig_acc, ax_acc = plt.subplots()
    fig_acc.suptitle("Train and Test Accuracy")
    ax_acc.plot(history["accuracy"] * 100, label="Train")
    ax_acc.plot(history["val_accuracy"] * 100, label="Test")
    fig_acc.supxlabel("Epoch")
    fig_acc.supylabel("Accuracy (%)")
    fig_acc.legend()
    ax_acc.grid()
    st.pyplot(fig_acc)


@st.cache_data
def plot_confusion_matrix(y_true, y_pred):
    fig_con, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    dist = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[i for i in range(10)]
    )

    fig_con.suptitle("Confusion Matrix")
    dist.plot(ax=ax)
    st.pyplot(fig_con)


@st.cache_data
def getYpredict(
    _model, X_test
):  # We didn't change any of the Y depending on the users so...
    y_pred = _model.predict(X_test)
    y_pred = y_pred.argmax(axis=1)
    return y_pred


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

# Visualize what happened at each step
# st.subheader("Filter Visualization and Feature Maps")
# st.write("Look at what the model extracted from and will detect in future images.")
# st.write(
#     """
#          To generate feature maps we need to understand ```model.layers``` API.
#          """
# )

# layer_outputs = [layer.output for layer in model.layers]
# input_image = X_test[index].reshape(-1, 28, 28, 1)
# activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(input_image)

# layer_names = [
#     "conv2d",
#     "max_pooling2d",
#     "conv2d_1",
#     "max_pooling2d_1",
#     "flatten",
#     "dense",
#     "dense_1",
#     "dense_2",
#     "dense_3",
# ]

# for layer_name, activation in zip(layer_names, activations):
#     if len(activation.shape) == 4:  # Check if the shape is 4D (i.e., feature maps)
#         n_features = activation.shape[-1]  # Number of feature maps in the layer
#         size = activation.shape[1]  # Size of each feature map
#         n_cols = int(np.sqrt(n_features))  # Number of columns for subplots
#         n_rows = int(np.ceil(n_features / n_cols))  # Number of rows for subplots

#         fig_feat = plt.figure(figsize=(n_cols, n_rows))
#         for i in range(n_features):
#             ax = fig_feat.add_subplot(n_rows, n_cols, i + 1)
#             ax.axis("off")
#             ax.imshow(activation[0, :, :, i], cmap="gray")  # Plot the ith feature map
#         fig_feat.suptitle(layer_name)

#         # Convert the Matplotlib figure to an image and display it in Streamlit
#         st.pyplot(fig_feat)
#         st.write(f"Layer: {layer_name}")
#         st.write(f"Number of Feature Maps: {n_features}")
#         st.write(f"Size of Each Feature Map: {size}x{size}")
#         st.write(f"Shape of Activation Output: {activation.shape}")
#         st.write("--------------------------")
# FOR DEBUGGING ONLY
# st.write([layer.name for layer in model.layers])
# st.write([layer.output for layer in model.layers])


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
st.header("Filter Visualization and Feature Maps")
st.write("Look at what the model extracted from and will detect in future images.")
# st.write(
#     """
#          To generate feature maps we need to understand ```model.layers``` API.
#          """
# )
st.write("From the above example:")
st.pyplot(fig_demo)

layer_outputs = [layer.output for layer in model.layers]
input_image = X_test[index].reshape(-1, 28, 28, 1)
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(input_image)

layer_names = [
    "conv2d",
    "max_pooling2d",
    "conv2d_1",
    "max_pooling2d_1",
    "flatten",
    "dense",
    "dense_1",
    "dense_2",
    "dense_3",
]

for layer_name, activation in zip(layer_names, activations):
    if len(activation.shape) == 4:  # Check if the shape is 4D (i.e., feature maps)
        n_features = activation.shape[-1]  # Number of feature maps in the layer
        size = activation.shape[1]  # Size of each feature map
        n_cols = int(np.sqrt(n_features))  # Number of columns for subplots
        n_rows = int(np.ceil(n_features / n_cols))  # Number of rows for subplots

        fig_feat = plt.figure(figsize=(n_cols, n_rows))
        for i in range(n_features):
            ax = fig_feat.add_subplot(n_rows, n_cols, i + 1)
            ax.axis("off")
            ax.imshow(activation[0, :, :, i], cmap="gray")  # Plot the ith feature map
        fig_feat.suptitle(layer_name)

        # Convert the Matplotlib figure to an image and display it in Streamlit
        st.pyplot(fig_feat)
        st.write(f"Layer: {layer_name}")
        st.write(f"Number of Feature Maps: {n_features}")
        st.write(f"Size of Each Feature Map: {size}x{size}")
        st.write(f"Shape of Activation Output: {activation.shape}")
        st.write("--------------------------")
# FOR DEBUGGING ONLY
# st.write([layer.name for layer in model.layers])
# st.write([layer.output for layer in model.layers])


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
history = load_history_csv(
    "https://github.com/Purinat33/CNN-Classroom/raw/master/history.csv", "history.csv"
)

history = history.rename(columns={"Unnamed: 0": "epoch"})

plot_history(history)

st.write(
    "[History Plotting Guide](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)"
)


st.write(
    """
         #### History Dataframe:
         """
)

history = history.rename(
    columns={
        "loss": "train_loss",
        "accuracy": "train_accuracy",
        "val_loss": "test_loss",
        "val_accuracy": "test_accuracy",
    }
)
st.dataframe(history, hide_index=True)

# Confusion Matrix
st.write(
    """
        ### Confusion Matrix:
        A confusion matrix presents a table layout of the different outcomes of the prediction and results of a classification problem and helps visualize its outcomes.

        It plots a table of all the predicted and actual values of a classifier. [Source](https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/)
         """
)

y_true = Y_test
X_test_re = X_test.reshape((10000, 28, 28, 1))
y_pred = getYpredict(model, X_test_re)
plot_confusion_matrix(y_true, y_pred)

st.write("--------------------------")
# feature_maps(model, X_test, index)
# st.write("--------------------------")
# st.write("layer.name")
# st.write([layer.name for layer in model.layers])

# st.write("--------------------------")
# st.write("layer.input")
# st.write([layer.input for layer in model.layers])

# st.write("--------------------------")
# st.write("layer.output")
# st.write([layer.output for layer in model.layers])


# st.write("--------------------------")


# layer_outputs = [layer.output for layer in model.layers]
# input_image = X_test[index].reshape(-1, 28, 28, 1)
# activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(input_image)

# layer_names = [
#     "conv2d",
#     "max_pooling2d",
#     "conv2d_1",
#     "max_pooling2d_1",
#     "flatten",
#     "dense",
#     "dense_1",
#     "dense_2",
#     "dense_3",
# ]

# for layer_name, activation in zip(layer_names, activations):
#     if len(activation.shape) == 4:  # Check if the shape is 4D (i.e., feature maps)
#         n_features = activation.shape[-1]  # Number of feature maps in the layer
#         size = activation.shape[1]  # Size of each feature map
#         n_cols = int(np.sqrt(n_features))  # Number of columns for subplots
#         n_rows = int(np.ceil(n_features / n_cols))  # Number of rows for subplots

#         fig_feat = plt.figure(figsize=(n_cols, n_rows))
#         for i in range(n_features):
#             ax = fig_feat.add_subplot(n_rows, n_cols, i + 1)
#             ax.axis("off")
#             ax.imshow(activation[0, :, :, i], cmap="gray")  # Plot the ith feature map
#         fig_feat.suptitle(layer_name)

#         # Convert the Matplotlib figure to an image and display it in Streamlit
#         st.pyplot(fig_feat)
#         st.write(f"Layer: {layer_name}")
#         st.write(f"Number of Feature Maps: {n_features}")
#         st.write(f"Size of Each Feature Map: {size}x{size}")
#         st.write(f"Shape of Activation Output: {activation.shape}")
#         st.write("--------------------------")
