import os
import zipfile

import tensorflow as tf
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems


class WordSenceDisambiguation(text_problems.Text2TextProblem):

    URL = "http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip"

    def generate_samples(self, data_dir, tmp_dir ,dataset_split):

        compressed_filename = os.path.basename((self.URL))
        download_path = generator_utils.maybe_download(tmp_dir, compressed_filename, self.URL)

        semcor_dir = os.path.join(tmp_dir, "semcor3.0")
        if not tf.gfile.Exists(semcor_dir):
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

