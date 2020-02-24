import tensorflow as tf

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_dim):
        self.generator, self.discriminator = model
        self.generator_optimizer, self.discriminator_optimizer = optimizers
        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.noise_dim = noise_dim
        self.measure = 0
        self.gamma = 0.5
        self.lambda_k = 0.001
        self.k_t = 0

    def calculate_loss(self, G, decode_G, R, decode_R):
        real_loss = tf.reduce_mean(tf.abs(R - decode_R))
        fake_loss = tf.reduce_mean(tf.abs(G - decode_G))

        d_loss = real_loss - self.k_t * fake_loss
        g_loss = fake_loss

        balance = self.gamma * real_loss - g_loss
        measure = real_loss + tf.abs(balance)

        self.k_t = tf.clip_by_value(self.k_t + self.lambda_k * balance, 0, 1)
        print('d_loss: {}, g_loss: {}, measure: {}'.format(d_loss, g_loss, measure))
        return d_loss, g_loss, measure

    def train_step(self, noise, images):
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            G = self.generator(noise, training=True)
            decode_G = self.discriminator(G, training=True)
            decode_R = self.discriminator(images, training=True)

            disc_loss, gen_loss, self.measure = self.calculate_loss(G, decode_G, images, decode_R)
        self.disc_loss(disc_loss)
        self.gen_loss(gen_loss)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

    def train(self, epoch,  pic):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()
        for (batch, images) in enumerate(self.train_dataset):
            noise = tf.random.normal([images.shape[0], self.noise_dim])
            self.train_step(noise, images)
            pic.add(self.measure)
            if (batch + 1) % 100 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}, measure: {}'.format(epoch, self.gen_loss.result(), self.disc_loss.result(), self.measure))