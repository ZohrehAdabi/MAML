from copy_model import copy_model
from eval_sine_test import eval_sine_test
from sine_generator import SinusoidGenerator 

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['dark_background'])
import matplotlib as mpl

colors = {0:'dodgerblue' , 1: 'tomato' , 2:'forestgreen'}

def eval_sinewave_for_test(model, sinusoid_generator=None, num_steps=(0, 1, 10), lr=0.01, plot=True):
    '''Evaluates how the sinewave addapts at dataset.
    
    The idea is to use the pretrained model as a weight initializer and
    try to fit the model on this new dataset.
    
    Args:
        model: Already trained model.
        sinusoid_generator: A sinusoidGenerator instance.
        num_steps: Number of training steps to be logged.
        lr: Learning rate used for training on the test data.
        plot: If plot is True than it plots how the curves are fitted along
            `num_steps`.
    
    Returns:
        The fit results. A list containing the loss, logits and step. For
        every step at `num_steps`.
    '''
    
    if sinusoid_generator is None:
        sinusoid_generator = SinusoidGenerator(K=10)
        
    # generate equally spaced samples for ploting
    x_test, y_test = sinusoid_generator.equally_spaced_samples(100)
    
    # batch used for training
    x, y = sinusoid_generator.batch()
    
    # copy model so we can use the same model multiple times
    copied_model = copy_model(model, x)
    
    # use SGD for this part of training as described in the paper
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    
    # run training and log fit results
    fit_res, w3_copied_model, weight_gradient = eval_sine_test(copied_model, optimizer, x, y, x_test, y_test, num_steps)
    
    # plot
    fig = plt.figure(figsize=(12, 5), tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    train, = ax.plot(x, y, '^')
    ground_truth, = ax.plot(x_test, y_test)
    plots = [train, ground_truth]
    legend = ['Training Points', 'True Function']
    i = 0
    for n, res, loss in fit_res:
        cur, = ax.plot(x_test, res[:, 0], '--', color=colors[i] )
        plots.append(cur)
        legend.append(f'After {n} Steps')
        i +=1%3
    ax.legend(plots, legend)
    ax.set_ylim(-5, 5)
    ax.set_xlim(-6, 6)
    if plot:
        plt.show()
    
    return fit_res, w3_copied_model, weight_gradient