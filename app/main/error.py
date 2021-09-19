# Custom errors for neural network responses
custom_errors = {
    'NeuralNetworkInternalError': {
        'message': "Cannot produce an image caption.",
        'status': 1001,
        'extra': "Caption for provided image cannot be produced."
    }
}
