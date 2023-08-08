import unittest
from unittest.mock import patch, Mock
import os
import sys
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image
from io import BytesIO

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'G:\My Drive\ICG_CNN_LSTM_Project\src\app.py', 'src'))
sys.path.insert(0, src_path)
from app import generate_caption, generate_caption_with_openai, idx_to_word

class TestGenerateCaption(unittest.TestCase):
    def setUp(self):
        self.features = {
            "image_1": (np.random.rand(4096),),
            "image_2": (np.random.rand(4096),),
        }

        self.tokenizer = Mock()
        self.tokenizer.texts_to_sequences.return_value = [[1, 2, 3, 4]]

    def test_idx_to_word_with_valid_integer(self):
        self.tokenizer.word_index = {"startseq": 1, "first": 2, "second": 3, "endseq": 4}
        word = idx_to_word(3, self.tokenizer)
        self.assertEqual(word, "second")

    def test_idx_to_word_with_invalid_integer(self):
        self.tokenizer.word_index = {"startseq": 1, "first": 2, "second": 3, "endseq": 4}
        word = idx_to_word(5, self.tokenizer)
        self.assertIsNone(word)

    @patch("openai.Completion.create")
    def test_generate_caption_with_openai(self, mock_openai_create):
        prompt = "Generate a caption for this image"
        mock_openai_create.return_value.choices[0].text.strip.return_value = "a generated caption"
        caption = generate_caption_with_openai(prompt)
        self.assertEqual(caption, "A Generated Caption")

    @patch("tensorflow.keras.models.load_model")
    def test_generate_caption(self, mock_load_model):
        # Create a mock model to be returned by load_model
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Create a custom mock for tokenizer.index_word
        mock_index_word = {
            0: "startseq",
            1: "first",
            2: "second",
            3: "endseq"
        }
        mock_tokenizer = Mock()
        mock_tokenizer.index_word = Mock(return_value=mock_index_word)

        # Set the side_effect for tokenizer.texts_to_sequences method
        mock_tokenizer.texts_to_sequences = Mock(return_value=[[1, 2, 3, 4]])

        # Set the side_effect for tokenizer.index_word.get method
        mock_tokenizer.index_word.get = Mock(side_effect=lambda x, default=None: mock_index_word[x] if default is None else default)

        prompt = "Generate a caption for this image:"
        new_caption = generate_caption("image_1.jpg", self.features, mock_tokenizer, 20, "Instagram")
        self.assertTrue(prompt in new_caption)

    def test_generate_caption_invalid_image_id(self):
        with self.assertRaises(KeyError):
            generate_caption("invalid_image_id.jpg", self.features, self.tokenizer, 20, "Instagram")

    @patch("tensorflow.keras.models.load_model")
    @patch("openai.Completion.create", side_effect=Exception("API Error"))
    def test_generate_caption_with_openai_exception(self, mock_openai_create, mock_load_model):
        # Create a mock model to be returned by load_model
        mock_model = Mock()
        mock_openai_create.return_value.choices[0].text.strip.return_value = "a generated caption"
        mock_load_model.return_value = mock_model

        # Create a custom mock for tokenizer.index_word
        mock_index_word = {
            0: "startseq",
            1: "first",
            2: "second",
            3: "endseq"
        }
        mock_tokenizer = Mock()
        mock_tokenizer.index_word = Mock(return_value=mock_index_word)

        # Set the side_effect for tokenizer.texts_to_sequences method
        mock_tokenizer.texts_to_sequences = Mock(return_value=[[1, 2, 3, 4]])

        # Set the side_effect for tokenizer.index_word.get method
        mock_tokenizer.index_word.get = Mock(side_effect=lambda x, default=None: mock_index_word[x] if default is None else default)

        prompt = "Generate a caption for this image:"
        with self.assertRaises(Exception):
            generate_caption("image_1.jpg", self.features, mock_tokenizer, 20, "Instagram")

if __name__ == "__main__":
    model_path = r"G:\My Drive\ICG_CNN_LSTM_Project\WORKING\best_model.h5"

    # Load the model
    model = load_model(model_path)

    BASE_DIR = r"G:\My Drive\ICG_CNN_LSTM_Project\BASE"
    WORKING_DIR = r"G:\My Drive\ICG_CNN_LSTM_Project\WORKING"

    tokenizer_path = r"G:\My Drive\ICG_CNN_LSTM_Project\BASE\tokenizer.pkl"
    mapping_path = r"G:\My Drive\ICG_CNN_LSTM_Project\WORKING\mapping.pkl"
    features_path = r"G:\My Drive\ICG_CNN_LSTM_Project\WORKING\features.pkl"

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
    unittest.main()
