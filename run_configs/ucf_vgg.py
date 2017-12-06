config = {
    'batch_size': 4,
    'class_limit': 20,
    'CPU_only': False,
    'early_stopping': True,
    'learning_rate': 5e-5,
    'nb_epoch': 1000,
    'optimizer': 'sgd',
    'patience': 30,
    'sequence_length': 30,
    'steps_per_epoch': 100,
    'validation_steps': 40,
    'verbose': True,
    'workers': 8,

    'tensorboard_callback': True,
    'early_stopper_callback': True,
    'checkpointer_callback': True,

    'run_label': 'vgg_rnn-30frames-sgd5e5-20-classes',
    'model': 'vgg_rnn',
    'dataset': 'ucf101',
    'model_path': 'data/checkpoints/vgg_rnn-30frames-sgd-20-classes.hdf5',
}
