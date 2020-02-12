save_model_weight
def save_model_weight(model, path):
    model.save_weights(path, save_format='tf')

# save_model_weight(neural_net, path="./net/neural_net_model")
# save_model_weight(maml, path="./maml/maml_model")