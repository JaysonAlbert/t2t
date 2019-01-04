import os
import tarfile

import tensorflow as tf
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators.imdb import SentimentIMDB
from tensor2tensor.utils import registry


@registry.register_problem
class FudanSentiment(SentimentIMDB):
    URL = 'https://raw.githubusercontent.com/JaysonAlbert/fudan_mtl_reviews/master/src/data/fudan-mtl-dataset.tar.gz'

    def doc_generator(self, imdb_dir, dataset, include_label=False):
        for filename in tf.gfile.Glob(os.path.join(imdb_dir,'*'+dataset)):
            for line in tf.gfile.Open(filename, 'rb'):
                try:
                    segments = line.decode('utf-8').strip().split('\t')
                    if len(segments) == 2:
                        yield segments[1], int(segments[0])
                except UnicodeDecodeError as e:
                    pass

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        compressed_filename = os.path.basename(self.URL)
        download_path = generator_utils.maybe_download(tmp_dir, compressed_filename,
                                                       self.URL)

        dir = os.path.join(tmp_dir, 'mtl-dataset')
        if not tf.gfile.Exists(dir):
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(tmp_dir)

        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "train" if train else "test"
        for doc, label in self.doc_generator(dir, dataset, include_label=True):
            yield {
                "inputs": doc,
                "label": int(label),
            }