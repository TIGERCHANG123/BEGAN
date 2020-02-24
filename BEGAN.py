from BEGAN_Block import *

class generator(tf.keras.Model):
  def __init__(self):
    super(generator, self).__init__()
    self.input_layer = generator_Input(shape=[8, 8, 512])

    self.middle_layer1 = generator_Middle(filters=256, strides=2)
    self.middle_layer2 = generator_Middle(filters=128, strides=2)
    self.middle_layer3 = generator_Middle(filters=64, strides=2)

    self.output_layer = generator_Output(image_depth=3, strides=1)
  def call(self, x):
    x = self.input_layer(x)
    x = self.middle_layer1(x)
    x = self.middle_layer2(x)
    x = self.middle_layer3(x)
    x = self.output_layer(x)
    return x

class discriminator(tf.keras.Model):
  def __init__(self, hidden):
    super(discriminator, self).__init__()

    self.input_layer = discriminator_Input(filters=256, strides=2)
    self.encoder_layer1 = discriminator_encoder(filters=128, strides=2)
    self.encoder_layer2 = discriminator_encoder(filters=64, strides=2)

    # self.middle = discriminator_middle(hidden, (64, 8, 8))

    self.decoder_layer1 = discriminator_decoder(filters=64, strides=2)
    self.decoder_layer2 = discriminator_decoder(filters=128, strides=2)
    self.decoder_layer3 = discriminator_decoder(filters=256, strides=2)

    self.decoder_output = discriminator_output(filters=3, strides=1)
  def call(self, x):
    x = self.input_layer(x)
    x = self.encoder_layer1(x)
    x = self.encoder_layer2(x)
    # x = self.middle(x)
    x = self.decoder_layer1(x)
    x = self.decoder_layer2(x)
    x = self.decoder_layer3(x)

    x = self.decoder_output(x)
    return x

def get_gan(noise_shape):
  Generator = generator()
  Generator.build(input_shape=(None, noise_shape))
  Generator.summary()
  Discriminator = discriminator(64)
  Discriminator.build(input_shape=(None, 64, 64, 3))
  Discriminator.summary()
  gen_name = 'dc_gan'
  return Generator, Discriminator, gen_name


