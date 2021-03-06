# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from BEGAN import get_gan
from show_pic import draw
from Train import train_one_epoch
from datasets.oxford_102_64 import oxford_102_flowers_dataset
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

ubuntu_root='/home/tigerc'
windows_root='D:/Automatic/SRTP/GAN'
root = ubuntu_root
temp_root = root+'/temp'

def main(continue_train, train_time):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    noise_dim = 100
    batch_size = 64

    generator_model, discriminator_model, model_name = get_gan(noise_shape=100)
    dataset = oxford_102_flowers_dataset(root, batch_size)
    model_dataset = model_name + '-' + dataset.name

    train_dataset = dataset.get_train_dataset()
    pic = draw(10, temp_root, model_dataset, train_time=train_time)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_path = temp_root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(genetator_optimizers=generator_optimizer, discriminator_optimizer=discriminator_optimizer ,
                               generator=generator_model, discriminator=discriminator_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint and continue_train:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    gen_loss = tf.keras.metrics.Mean(name='gen_loss')
    disc_loss = tf.keras.metrics.Mean(name='disc_loss')

    train = train_one_epoch(model=[generator_model, discriminator_model], train_dataset=train_dataset,
              optimizers=[generator_optimizer, discriminator_optimizer], metrics=[gen_loss, disc_loss], noise_dim=noise_dim)

    for epoch in range(1000):
        train.train(epoch=epoch, pic=pic)
        pic.show()
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
        pic.save_created_pic(generator_model, 8, noise_dim, epoch)
    pic.show_created_pic(generator_model, 8, noise_dim)
    return

if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # # tf.keras.backend.set_floatx('float64')

    main(continue_train=False, train_time=0)