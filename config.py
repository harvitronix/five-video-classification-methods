config = {
    'batch_size': 16,
    'early_stopping': True,
    'learning_rate': 1e-2,
    'nb_epoch': 1000,
    'optimizer': 'sgd',
    'patience': 5,
    'sequence_length': 10,
    'steps_per_epoch': 100,
    'validation_steps': 20,
    'verbose': True,
    'workers': 1,

    'tensorboard_callback': True,
    'early_stopper_callback': True,
    'checkpointer_callback': False,
    
    'models': {
        'c3d': {
            'paths': [
                {
                    'preprocessing': 'images',
                    'input_shape': (80, 80, 3),
                },
            ],
        },
        'vgg_rnn': {
            'paths': [
                {
                    'preprocessing': 'images',
                    'input_shape': (80, 80, 3),
                },
            ],
        },
        'conv3d': {
            'paths': [
                {
                    'preprocessing': 'images',
                    'input_shape': (80, 80, 3),
                },
            ],
        },
        'lrcn': {
            'paths': [
                {
                    'preprocessing': 'images',
                    'input_shape': (80, 80, 3),
                },
            ],
        },
    },

    'datasets': {
        'ucf101': {
            'nb_classes': 10,
        },
    },
}
