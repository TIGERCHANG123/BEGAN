import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from time import sleep
import os
import tensorflow as tf

class draw:
  batch_list = []
  loss_list = []
  acc_list = []
  i = 0
  def __init__(self, pic_size, root, model_dataset, train_time):
    rcParams['figure.figsize']=pic_size, pic_size
    self.pic_path = root + '/temp_pic/' + model_dataset
    self.pic_save_path = root + '/temp_pic_save/' + model_dataset
    self.generated_pic_path = root + '/generated_pic/' + model_dataset

    if not (os.path.exists(self.pic_path)):
        os.makedirs(self.pic_path)
    if not (os.path.exists(self.pic_save_path)):
        os.makedirs(self.pic_save_path)
    if not (os.path.exists(self.generated_pic_path)):
        os.makedirs(self.generated_pic_path)

    self.fig = plt.figure(figsize=(12, 4))
    self.train_time = train_time
    self.batch_list = []
    self.train_loss_list = []
    self.i = 0
  def add(self, train_loss):
    if len(self.batch_list) != 0:
      self.i = self.batch_list[-1] + 1
    else:
      self.i = self.i+1
    self.batch_list.append(self.i)
    self.train_loss_list.append(train_loss)
  def close(self, time):
      sleep(time)
      plt.close()
  def show(self):
    file_path = self.pic_path
    plt.clf()
    ax1 = self.fig
    ax1.plot(self.batch_list, self.train_loss_list, label = 'train loss', color = 'red')
    bbox_props = dict(boxstyle='round',fc='w', ec='k',lw=1)
    ax1.annotate("%s" % self.train_loss_list[-1], xy=(self.i, self.train_loss_list[-1]), xytext=(-20, -20), textcoords='offset points', bbox=bbox_props)
    ax1.set(xlabel='batches',ylabel='loss', title = 'gen_loss')

    plt.savefig(file_path+'/{}_{}.png'.format(self.train_time, str(self.i)))
    # thread1 = Thread(target=self.close, args=(1,))
    # thread1.start()
    # plt.show()

  def show_image(self, image):
    plt.imshow(image)
    plt.show()

  def show_created_pic(self, generator, pic_num, noise_dim):
    x = tf.convert_to_tensor(np.random.rand(pic_num, noise_dim))
    y = generator(x)
    y = tf.squeeze(y)
    y = (y + 1) / 2 * 255
    y = tf.cast(y, tf.uint8)
    for i in range(pic_num):
      plt.subplot(1, pic_num, i + 1)
      plt.imshow(y[i].numpy())
      plt.axis('off')
      plt.tight_layout()
    plt.show()
    return

  def save_created_pic(self, generator, pic_num, noise_dim, epoch):
    x = tf.convert_to_tensor(np.random.rand(pic_num, noise_dim))
    y = generator(x)
    y=tf.squeeze(y)
    y = (y + 1) / 2 * 255
    y = tf.cast(y, tf.uint8)
    for i in range(pic_num):
      plt.imsave(self.generated_pic_path+'/{}_{}_{}.png'.format(self.train_time, epoch, i), y[i].numpy())
    return