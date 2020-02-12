
def load_model_weight(model_calss, path, x):
    model = model_calss
    model.load_weights(path)
    model.call(tf.convert_to_tensor(x))
    return model
# sinusoid_generator = SinusoidGenerator(K=10)
# x, _ = sinusoid_generator.batch()
# neural_net = load_model_weight(SineModel(), path="./net/neural_net_model", x=x)
# maml = load_model_weight(SineModel(), path="./maml/maml_model", x=x)

# print(neural_net.out.get_weights()[0].shape)
# print(maml.out.get_weights()[0].shape)