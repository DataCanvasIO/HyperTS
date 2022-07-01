import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from hyperts.framework.dl import layers

class Test_DL_Layers():

    def test_multicolembedding_layers(self):
        data = np.random.randint(100, size=(4, 3, 2))

        model = tf.keras.Sequential()
        model.add(layers.MultiColEmbedding(input_dims=[100, 100], output_dims=[4, 4]))

        output = model.predict(data)

        assert output.shape == (4, 3, 8)

    def test_weightedattention_layers(self):
        data = tf.reshape(tf.range(0, 24), shape=(4, 3, 2)) / 24

        output = layers.WeightedAttention(timesteps=3)(data)

        assert output.shape == (4, 3, 2)

    def test_feedforwardattention_layers(self):
        data = tf.reshape(tf.range(0, 24), shape=(4, 3, 2)) / 24

        output1 = layers.FeedForwardAttention(return_sequences=True)(data)
        output2 = layers.FeedForwardAttention(return_sequences=False)(data)

        assert output1.shape == (4, 3, 2)
        assert output2.shape == (4, 2)

    def test_autoregressive_layers(self):
        data = tf.reshape(tf.range(0, 24), shape=(4, 3, 2)) / 24

        output = layers.AutoRegressive(order=1, nb_variables=2)(data)

        assert output.shape == (4, 2)

    def test_highway_layers(self):
        data = tf.reshape(tf.range(0, 24), shape=(4, 3, 2)) / 24

        output = layers.Highway(nb_variables=2)(data)

        assert output.shape == (4, 2)

    def test_time2vec_layers(self):
        data = tf.reshape(tf.range(0, 24), shape=(4, 3, 2)) / 24

        output = layers.Time2Vec(kernel_size=2)(data)

        assert output.shape == (4, 3, 4)

    def test_revin_layers(self):
        data = tf.reshape(tf.range(0, 24), shape=(4, 3, 2)) / 24

        revin = layers.RevInstanceNormalization()
        output1 = revin(data, mode='norm')
        output2 = revin(output1, mode='denorm')

        assert output1.shape == (4, 3, 2)
        assert output2.shape == (4, 3, 2)

    def test_identify_layers(self):
        data = tf.reshape(tf.range(0, 24), shape=(4, 3, 2)) / 24

        output = layers.Identity()(data)

        assert output.shape == (4, 3, 2)

    def test_shortcut_layers(self):
        data = tf.reshape(tf.range(0, 24), shape=(4, 3, 2)) / 24

        output = layers.Shortcut(filters=8)(data)

        assert output.shape == (4, 3, 8)

    def test_inceptionblock(self):
        data = tf.random.normal(shape=(4, 8, 2))

        kernel_size_list = (1, 2, 3, 5, 12)
        kernel_size_list = list(filter(lambda x: x < 8, kernel_size_list))

        output = layers.InceptionBlock(filters=8,
                                       kernel_size_list=kernel_size_list,
                                       bottleneck_size=8)(data)

        assert output.shape == (4, 8, 40)