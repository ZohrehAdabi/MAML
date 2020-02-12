from sine_generator import SinusoidGenerator
def generate_dataset(K, train_size=20000, test_size=10):
    '''Generate train and test dataset.
    
    A dataset is composed of SinusoidGenerators that are able to provide
    a batch (`K`) elements at a time.
    '''
    def _generate_dataset(size):
        return [SinusoidGenerator(K=K) for _ in range(size)]
    return _generate_dataset(train_size), _generate_dataset(test_size) 

# train_ds, test_ds = generate_dataset(K=10, test_size=25)
# num_batches = len(train_ds)
# train_ds[0]