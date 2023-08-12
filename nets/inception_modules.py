import tensorflow as tf
from tensorflow.python.keras import layers, Model, Sequential

from nets.utils import getConvBlock


class getInceptionTest(Model):
    def __init__(self, inc):
        super().__init__()
        self.b1 = layers.MaxPool2D((2, 2))
        self.b2 = getConvBlock(inc, 1, True, st=2)
        self.b3 = Sequential([
            getConvBlock(inc, 1, True),
            getConvBlock(inc, 1, True, st=2)
        ])

    def call(self, inputs):
        return tf.concat([self.b1(inputs), self.b2(inputs), self.b3(inputs)], axis=-1)



