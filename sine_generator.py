import numpy as np
import matplotlib.pyplot as  plt
# plt.style.use(['dark_background'])

class SinusoidGenerator():
    '''
        Sinusoid Generator.
        
        p(T) is continuous, where the amplitude varies within [0.1, 5.0]
        and the phase varies within [0, π].
        
        This abstraction is the basically the same defined at:
        https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0  
    '''
    def __init__(self, K=10, amplitude=None, phase=None):
        '''
        Args:
            K: batch size. Number of values sampled at every batch.
            amplitude: Sine wave amplitude. If None is uniformly sampled from
                the [0.1, 5.0] interval.
            pahse: Sine wave phase. If None is uniformly sampled from the [0, π]
                interval.
        '''
        self.K = K
        self.amplitude = amplitude if amplitude else np.random.uniform(0.1, 5.0)
        self.phase = phase if amplitude else np.random.uniform(-np.pi, np.pi)
        self.sampled_points = None
        self.x = self._sample_x()
        
    def _sample_x(self):
        return np.random.uniform(-5, 5, self.K)
    
    def f(self, x):
        '''Sinewave function.'''
        return self.amplitude * np.sin(x - self.phase)

    def batch(self, x = None, force_new=False):
        '''Returns a batch of size K.
        
        It also changes the shape of `x` to add a batch dimension to it.
        
        Args:
            x: Batch data, if given `y` is generated based on this data.
                Usually it is None. If None `self.x` is used.
            force_new: Instead of using `x` argument the batch data is
                uniformly sampled.
        
        '''
        if x is None:
            if force_new:
                x = self._sample_x()
            else:
                x = self.x
        y = self.f(x)
        return x[:, None], y[:, None]
    
    def equally_spaced_samples(self, K=None):
        '''Returns `K` equally spaced samples.'''
        if K is None:
            K = self.K
        return self.batch(x=np.linspace(-5, 5, K))
        
        
    def plot_sine(self, sine_generator, *args, **kwargs):
        '''Plot helper.'''
        fig = plt.figure(figsize=(6,4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Sinusoid examples')
        colors = {0:'dodgerblue' , 1: 'tomato' , 2:'forestgreen'}
        for i in range(3):
            data = sine_generator(K=10)
            x, y = data.batch()
            ax.plot(x, y,'^', color=colors[i])
            x, y = data.equally_spaced_samples(100)
            ax.plot(x, y, color=colors[i])
    def plot(self, data, fig, color,*args, **kwargs):
        '''Plot helper.'''
        # fig = plt.figure(figsize=(6,4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Sinusoid examples')
        x, y = data.batch()
        ax.plot(x, y,'^', color=color)
        x, y = data.equally_spaced_samples(100)
        ax.plot(x, y, color=color)
        
        
        


# fig = plt.figure(figsize=(6,4), dpi=100)
# ax = fig.add_subplot(111)
# ax.set_title('Sinusoid examples')
# colors = {0:'dodgerblue' , 1: 'tomato' , 2:'forestgreen'}
# for i in range(3):

#     plot(SinusoidGenerator(K=10),colors[i])
# plt.show()