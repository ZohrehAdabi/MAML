plot_model_comparison_to_average
def plot_model_comparison_to_average(model, ds, model_name='neural network', K=10):
    '''Compare model to average.
    
    Computes mean of training sine waves actual `y` and compare to
    the model's prediction to a new sine wave, the intuition is that
    these two plots should be similar.
    '''
    sinu_generator = SinusoidGenerator(K=K)
    
    # calculate average prediction
    avg_pred = []
    for i, sinusoid_generator in enumerate(ds):
        _ , y = sinusoid_generator.equally_spaced_samples()
        avg_pred.append(y)
    
    x, _ = sinu_generator.equally_spaced_samples()    
    avg_plot, = plt.plot(x, np.mean(avg_pred, axis=0),'--',color=colors[2])

    # calculate model prediction
    model_pred = model.call(tf.convert_to_tensor(x))
    model_plot, = plt.plot(x, model_pred.numpy(), color=colors[0])
    
    # plot
    plt.legend([avg_plot, model_plot], ['Average', model_name])
    plt.show()