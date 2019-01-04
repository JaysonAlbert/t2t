import tensorflow as tf
import tensorflow_hub as hub


class EmbeddingServer:

    def __init__(self, hub_model="https://tfhub.dev/google/elmo/2", trainable=False, signature="default"):
        self.trainable=trainable
        self.signature = signature
        self._sess = tf.Session()
        self._model = hub.Module(hub_model, trainable=trainable)
        self._input = tf.placeholder(tf.string,[None])
        self.res = self._model(self._input, signature=self.signature, as_dict=True)['default']

        self._sess.run([tf.initialize_all_tables(), tf.initialize_all_variables()])

    def embedding(self, sentence, signature="default"):
        if isinstance(sentence, str):
            sentence = [sentence]
        return self._sess.run(self.res, feed_dict={self._input: sentence})[0]

    def __del__(self):
        self._sess.close()
