import re

from gutenberg import acquire
from gutenberg import cleanup

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry, metrics
from tensor2tensor.data_generators.imdb import SentimentIMDB
from tensor2tensor.layers import modalities
import tensorflow as tf
import pandas as pd
from tensor2tensor.models.lstm import lstm_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.data_generators import generator_utils
import zipfile
import os


class WordSenceDisambiguation(text_problems.Text2TextProblem):

    URL = "http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip"

    def generate_samples(self, data_dir, tmp_dir ,dataset_split):

        compressed_filename = os.path.basename((self.URL))
        download_path = generator_utils.maybe_download(tmp_dir, compressed_filename, self.URL)

        semcor_dir = os.path.join(tmp_dir, "semcor3.0")
        if not tf.gfile.Exists(semcor_dir):
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

