import openai
import re
import os
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from PIL import Image
model_path = r"Download and upload the model file from the link given in README.md file and use that path here."

openai.api_key = "your_api_key"

def idx_to_word(integer, tokenizer):
    """
    Convert an integer to its corresponding word in the tokenizer's word index.

    Args:
        integer (int): The integer to convert.
        tokenizer (Tokenizer): The Tokenizer object containing the word index.

    Returns:
        str or None: The corresponding word if found, otherwise None.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption_with_openai(prompt):
    """
    Generate a caption using the OpenAI API.

    Args:
        prompt (str): The prompt for the OpenAI API to generate a caption.

    Returns:
        str: The generated caption.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",  # Replace with the appropriate model name
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    caption = response.choices[0].text.strip()

    # Filter out random character strings
    random_chars = re.findall(r'[A-Za-z0-9_]{10,}', caption)
    for random_char in random_chars:
        if "_" not in random_char:
            caption = caption.replace(random_char, '')

    # Capitalize every first word in the caption
    caption = caption.title()

    return caption


def generate_caption(image_id, features, tokenizer, max_length, platform):
    """
    Generate a caption for an image using a trained model and OpenAI API.

    Args:
        image_id (str): The ID or name of the image.
        features (dict): A dictionary containing image features extracted by the VGG16 model.
        tokenizer (Tokenizer): The Tokenizer object used to preprocess the captions.
        max_length (int): The maximum length of the caption sequence.
        platform (str): The social media platform for which the caption will be generated.

    Returns:
        str: The generated caption for the image and platform.
    """
    image_id = image_id.split('.')[0]  # Remove the file extension
    image_features = features[image_id][0]
    max_length = 35
    initial_caption = f"{platform} caption for this photo will be: startseq"
    sequence = tokenizer.texts_to_sequences([initial_caption])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    model = load_model(model_path)

    prediction = model.predict([np.expand_dims(image_features, axis=0), sequence], verbose=0)
    prediction = np.argmax(prediction)
    word = tokenizer.index_word.get(prediction, "")

    while word != "endseq":
        initial_caption += " " + word
        sequence = tokenizer.texts_to_sequences([initial_caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        prediction = model.predict([np.expand_dims(image_features, axis=0), sequence], verbose=0)
        prediction = np.argmax(prediction)
        word = tokenizer.index_word.get(prediction, "")

    # Generate new caption using OpenAI API
    prompt = f"Generate a {platform} caption based on this image: {image_id}"
    new_caption = generate_caption_with_openai(prompt)

    return new_caption


def main():
    """
    Main function to run the Streamlit application for generating captions for images.
    """
    # Add a background color or image
    st.markdown(
        """
        <style>
        body {
            background-color: #F5F5DC;  /* Set your desired background color */
        }
        .caption-box {
            font-family: 'Times New Roman', Times, serif;
            font-weight: bold;
            border: 2px solid black;
            padding: 10px;
            display: inline-block;
            border-radius: 10px; /* Set border radius to make the edges rounded */
            background-color: #FFFFFF; /* Set background color of the caption box */
            color: black; /* Set text color to black */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("CAPTIVATE")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        platforms = {
            "Instagram": "Generate a Instagram Caption based on this image: ",
            "Youtube": "Generate a Youtube Caption based on this image: ",
            "Facebook": "Generate a Facebook Caption based on this image: "
        }

        selected_platform = st.selectbox("Select Social Media Platform", list(platforms.keys()))

        # Generate caption for the uploaded image
        if st.button("Generate Caption"):
            # Preprocess the image
            image = image.convert("RGB")
            image = image.resize((224, 224))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)

            # Extract features from the image using VGG16 model
            image_features = vgg_model.predict(image_array)

            # Generate caption using the trained model
            prompt = "Generate an Instagram caption for this image:"
            new_caption = generate_caption_with_openai(prompt)

            st.markdown('<div class="caption-box">Captivate Suggests:<br/>{}</div>'.format(new_caption), unsafe_allow_html=True)


if __name__ == "__main__":

    model_path = r"G:\My Drive\ICG_CNN_LSTM_Project\WORKING\best_model.h5"

    # Load the model
    model = load_model(model_path)

    BASE_DIR = r"G:\My Drive\ICG_CNN_LSTM_Project\BASE"
    WORKING_DIR = r"G:\My Drive\ICG_CNN_LSTM_Project\WORKING"

    tokenizer_path = r"Download and upload the tokenizer file from the link given in README.md file and use that path here."
    mapping_path = r"Download and upload the mapping file from the link given in README.md file and use that path here."
    features_path = r"Download and upload the features file from the link given in README.md file and use that path here."

    # Load the tokenizer and mapping
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)

    # Load the features
    with open(features_path, "rb") as f:
        features = pickle.load(f)

    features = {k.split('.')[0]: v for k, v in features.items()}

    # Load the VGG16 model
    vgg_model = VGG16()
    vgg_model = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)

main()
