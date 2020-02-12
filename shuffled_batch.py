
def shuffled_batch(train_ds, num_batches=20000, batch_size=10):
    x_train,y_train =np.array([]), np.array([])
    for i, sinusoid_generator in enumerate(train_ds):
            x , y = sinusoid_generator.batch()
            
            x_train = np.append(x_train, x)
            y_train = np.append(y_train, y)
    permute = np.random.permutation(x_train.shape[0])
    for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = (batch_index + 1) * batch_size
            x_batch = x_train[permute[start: end]]
            y_batch = y_train[permute[start: end]]
            yield x_batch, y_batch
# i=0
# for x, y in shuffled_batch(train_ds, num_batches):
#   print(i,x,y)
#   i+=1