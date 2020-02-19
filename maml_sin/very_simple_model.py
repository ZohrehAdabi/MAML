
from tensorflow import keras
# import tensorflow.keras.backend  as K
import tensorflow as tf
class SimpleSineModel(keras.Model):
    def __init__(self):
        super(SimpleSineModel, self).__init__()
        self.hidden1 = keras.layers.Dense(1, input_shape=(1,))
        self.out = keras.layers.Dense(1)
        
    def call(self, x):
        x = tf.sin(self.hidden1(x))
        x = self.out(x)
        return x