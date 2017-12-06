config = {
    'batch_size': 4,
    'class_limit': None,
    'CPU_only': False,
    'early_stopping': True,
    'learning_rate': 1e-3,
    'nb_epoch': 1000,
    'optimizer': 'sgd',
    'patience': 10,
    'sequence_length': 10,
    'steps_per_epoch': 100,
    'validation_steps': 40,
    'verbose': True,
    'workers': 8,

    'tensorboard_callback': True,
    'early_stopper_callback': True,
    'checkpointer_callback': False,

    'run_label': 'synth_vgg_rnn-1e-3',
    'model': 'vgg_rnn',
    'dataset': 'synthetic_boxes',
    'model_path': None,
}
