import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class generator_Input(tf.keras.Model):
  def __init__(self, shape):
    super(generator_Input, self).__init__()
    self.dense = layers.Dense(shape[0] * shape[1] * shape[2], use_bias=False)
    self.reshape = layers.Reshape(shape)
    self.bn = layers.BatchNormalization(momentum=0.9)
    self.relu = tf.keras.layers.ReLU()
  def call(self, x):
    x = self.dense(x)
    x = self.reshape(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class generator_Middle(tf.keras.Model):
  def __init__(self, filters, strides):
      super(generator_Middle, self).__init__()
      self.conv = layers.Conv2DTranspose(filters, (5, 5), strides=strides, padding='same', use_bias=False)
      self.bn = layers.BatchNormalization(momentum=0.9)
      self.relu = tf.keras.layers.ReLU()
  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.relu(x)
      return x

class generator_Output(tf.keras.Model):
  def __init__(self, image_depth, strides):
    super(generator_Output, self).__init__()
    self.conv = layers.Conv2DTranspose(image_depth, (5, 5), strides=strides, padding='same', use_bias=False, activation='tanh')
    # self.actv = layers.Activation(activation='tanh')
  def call(self, x):
    x = self.conv(x)
    # x = self.actv(x)
    return x

class discriminator_Input(tf.keras.Model):
  def __init__(self, filters, strides):
    super(discriminator_Input, self).__init__()
    self.conv = keras.layers.Conv2D(filters, kernel_size=5, strides=strides, padding="same")
    self.leakyRelu = keras.layers.LeakyReLU(alpha=0.2)
    self.dropout = keras.layers.Dropout(0.3)

  def call(self, x):
    x = self.conv(x)
    x = self.leakyRelu(x)
    x = self.dropout(x)
    return x

class discriminator_encoder(tf.keras.Model):
  def __init__(self, filters, strides):
      super(discriminator_encoder, self).__init__()
      self.conv = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same")
      self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)
      self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.2)
      self.dropout = tf.keras.layers.Dropout(0.3)

  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.leakyRelu(x)
      x = self.dropout(x)
      return x

class discriminator_middle(tf.keras.Model):
    def __init__(self, hidden, shape):
        super(discriminator_middle, self).__init__()
        self.flatten = layers.Flatten()
        self.encoder_embedding = layers.Dense(units=hidden)
        self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.decoder_embedding = layers.Dense(units=shape[0]*shape[1]*shape[2])
        self.reshape = layers.Reshape(target_shape=shape)
    def call(self, x):
        x = self.flatten(x)
        x = self.encoder_embedding(x)
        x = self.leakyRelu(x)
        x = self.decoder_embedding(x)
        x = self.reshape(x)
        return x

class discriminator_decoder(tf.keras.Model):
  def __init__(self, filters, strides):
      super(discriminator_decoder, self).__init__()
      self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=strides, padding="same")
      self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)
      self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.2)
      self.dropout = tf.keras.layers.Dropout(0.3)

  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.leakyRelu(x)
      x = self.dropout(x)
      return x

class discriminator_output(tf.keras.Model):
  def __init__(self, filters, strides):
    super(discriminator_output, self).__init__()
    self.conv = keras.layers.Conv2D(filters, kernel_size=5, strides=strides, padding="same")

  def call(self, x):
    x = self.conv(x)
    return x









