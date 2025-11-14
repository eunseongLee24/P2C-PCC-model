import tensorflow as tf
from tensorflow.keras import layers, backend as K

class P2C_PCC(layers.Layer):
    """Positive-to-Convolution and Parallel Convolution Component (P2C-PCC)"""

    def __init__(self, filters=100, kernel_size=3, strides=1, padding='same', **kwargs):
        super(P2C_PCC, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Positive branch
        self.Wp = self.add_weight(
            name='Wp',
            shape=(self.kernel_size, input_dim, self.filters),
            initializer='uniform',
            trainable=True
        )
        self.bp = self.add_weight(
            name='bp',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )

        # Negative branch
        self.Wn = self.add_weight(
            name='Wn',
            shape=(self.kernel_size, input_dim, self.filters),
            initializer='uniform',
            trainable=True
        )
        self.bn = self.add_weight(
            name='bn',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # Positive path
        pos_att = tf.nn.conv1d(inputs, self.Wp, stride=self.strides, padding='SAME')
        pos = K.relu(K.sigmoid(pos_att) + self.bp)

        # Negative path
        neg_att = tf.nn.conv1d(inputs, self.Wn, stride=self.strides, padding='SAME')
        neg = K.minimum(K.sigmoid(neg_att) + self.bn, 0)

        # Merge
        return K.concatenate([pos, neg], axis=-1)

    def get_config(self):
        config = super(P2C_PCC, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config
