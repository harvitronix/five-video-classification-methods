config = {
    'batch_size': 4,
    'class_limit': None,
    'CPU_only': False,
    'learning_rate': 1e-2,
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

    'run_label': 'synth_conv_rnn-1e-2',
    'model': 'conv3d',
    'dataset': 'synthetic_boxes',
    'model_path': None,
}
