config = {
    'batch_size': 8,
    'class_limit': 10,
    'CPU_only': False,
    'early_stopping': True,
    'learning_rate': 1e-4,
    'nb_epoch': 100,
    'optimizer': 'adam',
    'patience': 5,
    'sequence_length': 40,
    'steps_per_epoch': 100,
    'validation_steps': 40,
    'verbose': True,
    'workers': 8,

    'tensorboard_callback': True,
    'early_stopper_callback': True,
    'checkpointer_callback': True,

    'run_label': 'conv3d-40frames-adam1e-4',
    'model': 'conv3d',
    'dataset': 'ucf101',
}
