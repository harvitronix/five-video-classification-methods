config = {
    'batch_size': 32,
    'early_stopping': True,
    'learning_rate': 1e-5,
    'nb_epoch': 1000,
    'optimizer': 'adam',
    'patience': 5,
    'sequence_length': 40,
    'steps_per_epoch': 1000,
    'validation_steps': 40,
    'verbose': True,
    'workers': 4,

    'tensorboard_callback': True,
    'early_stopper_callback': True,
    'checkpointer_callback': True,
    
    'models': {
        'c3d': {
            'paths': [
                {
                    'preprocessing_steps': [],
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
