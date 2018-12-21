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


class LSTMRealModality(modalities.RealL2LossModality):

    @property
    def name(self):
        return "lstm_real_modality_%d_%d" % (self._vocab_size,
                                               self._body_input_depth)

    def top(self, body_output, _):
        with tf.variable_scope(self.name):
            x = body_output
            # x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            shape = common_layers.shape_list(x)
            x = tf.slice(x,[0,shape[1]-1,0,0],[-1,1,-1,-1])
            res = tf.layers.dense(x, self._vocab_size)
            return tf.expand_dims(res, 3)


def prepare_data(path):
    df = pd.read_json(path)
    df['inputs'] = df['spans'].apply(lambda x:', '.join(x))
    df['targets'] = df['sentiment score']
    return df[['inputs', 'targets']]


@registry.register_problem
class SemEvalSentiment(SentimentIMDB):

    @property
    def is_generate_per_split(self):
        return False

    def generate_samples(self, data_dir, temp_dir, dataset_split):
        df = prepare_data(temp_dir)
        for index, row in df.iterrows():
            yield {
                "inputs": row["inputs"],
                "label": row["targets"]
            }

    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.FixedLenFeature([1], tf.float32),
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC, metrics.Metrics.ROC_AUC,
            metrics.Metrics.RMSE, metrics.Metrics.ABS_ERR,
        ]


    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.modality = {
            "inputs": modalities.SymbolModality,
            "targets": LSTMRealModality
        }

        p.vocab_size = {
            "inputs":self._encoders["inputs"].vocab_size,
            "targets": 1
        }


@registry.register_hparams
def semeval_lstm():
    hparams = lstm_attention()
    hparams.batch_size = 32

    return hparams