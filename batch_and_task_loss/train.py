import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as keras_backend
tf.keras.backend.set_floatx('float64')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['dark_background'])
import matplotlib as mpl
import time
device_name = tf.test.gpu_device_name()
np.random.seed(333)

from shuffled_batch import shuffled_batch

def loss_function(pred_y, y):
  return keras_backend.mean(keras.losses.mean_squared_error(y, pred_y))

def np_to_tensor(list_of_numpy_objs):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objs)
    

def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model.call(x)
    mse = loss_fn(y, logits)
    return mse, logits


def compute_gradients(model, x, y, loss_fn=loss_function):
    with tf.GradientTape() as tape:
        loss, _ = compute_loss(model, x, y, loss_fn)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

    
def train_batch(x, y, model, optimizer):
    tensor_x, tensor_y = np_to_tensor((x, y))
    gradients, loss = compute_gradients(model, tensor_x, tensor_y)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    return loss


def train_model(model, dataset, epochs=1, lr=0.01, log_steps=1000):
    # model = SineModel()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        losses = []
        inner_loop_loss = []
        total_loss = 0
        start = time.time()
        for i, sinusoid_generator in enumerate(dataset):
            x, y = sinusoid_generator.batch()
            model.call(x) # run forward pass to initialize weights???
            # print(type(x))
            loss = train_batch(x, y, model, optimizer)
            inner_loop_loss.append(loss)
            total_loss += loss
            curr_loss = total_loss / (i + 1.0)
            losses.append(curr_loss)
           
            
            if i % log_steps == 0 and i >= 0:
                print('Step {}: loss = {}, Time to run {} steps = {:.2f} seconds'.format(
                    i, curr_loss, log_steps, time.time() - start))
                start = time.time()
        fig = plt.figure(figsize=(12, 5), tight_layout=True)
        ax = fig.add_subplot(1,1,1)
        ax.plot(losses)
        ax.set_title('Loss Vs Time steps')
        plt.show()
        # print(losses)
    return model, losses, inner_loop_loss

def train_model_shuffled(model, dataset, epochs=1, lr=0.01, log_steps=1000):
    # model = SineModel()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        losses = []
        total_loss = 0
        start = time.time()
        i=0
        for x, y in shuffled_batch(dataset,  len(dataset)): #num_batches
            x=np.array(x, ndmin=2).T
            y=np.array(y, ndmin=2).T
            # print(type(x), x.shape)
            loss = train_batch(x, y, model, optimizer)
            total_loss += loss
            curr_loss = total_loss / (i + 1.0)
            losses.append(curr_loss)
            
            
            if i % log_steps == 0 and i >= 0:
                print('Step {}: loss = {}, Time to run {} steps = {:.2f} seconds'.format(
                    i, curr_loss, log_steps, time.time() - start))
                start = time.time()
            i+=1
        fig = plt.figure(figsize=(12, 5), tight_layout=True)
        ax = fig.add_subplot(1,1,1)
        ax.plot(losses)
        ax.set_title('Loss Vs Time steps')
        plt.show()
#         print(losses)
    return model