"""
Model training starts here. Configure the run in config.py, then run this
file to train your model on your dataset.
"""
def train(model, generators, run_label):
    train_gen, val_gen = generators

    if not run_label:
        run_label = str(time.time())

    callbacks = []
    if config['tensorboard_callback']:
        callbacks.append(TensorBoard(log_dir=os.path.join('data', 'logs', run_label)))
    if config['early_stopper_callback']:
        callbacks.append(EarlyStopping(patience=config['patience']))
    if config['checkpointer_callback']:
        callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join('data', 'checkpoints', run_label + '.hdf5'),
                verbose=config['verbose'],
                save_best_only=True
            )
        )

    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=config['steps_per_epoch'],
        epochs=config['nb_epoch'],
        verbose=config['verbose'],
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=config['validation_steps'],
        workers=config['workers'])

    return model


def main(config):
    """Based on the configuration files, load a model and a dataset and train."""
    model_name = config['model']
    dataset_name = config['dataset']
    model_path = config['model_path']
    paths = models_config['models'][model_name]['paths']
    input_shapes = [x['input_shape'] for x in paths]
    preprocessing_steps = [x['preprocessing'] for x in paths]
    if config['class_limit']:
        nb_classes = config['class_limit']
    else:
        nb_classes = datasets_config['datasets'][dataset]['nb_classes']

    # Get the model.
    if model_path:
        model = load_model(model_path)
    else: 
        model = get_model(nb_classes, model_name, config['sequence_length'],
                          config['optimizer'], config['learning_rate'], input_shapes,
                          config['verbose'])

    # Get the data generators.
    generators = get_generators(dataset_name, config['sequence_length'], nb_classes,
                                input_shapes, preprocessing_steps,
                                config['batch_size'])
    
    train(model, generators, config['run_label'])
        
if __name__ == '__main__':
    import argparse
    import importlib
    import os.path
    parser = argparse.ArgumentParser(description='Train a classifier.')
    parser.add_argument('config', action='store',
                        help='The filename of the run config.')
    args = parser.parse_args()
    
    # Get the config...
    imp = importlib.import_module('run_configs.' + args.config)
    config = imp.config
    
    # Hack to use CPU (used if model can't fit into GPU memory)
    # See https://github.com/fchollet/keras/issues/4613
    if config['CPU_only']:
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # As crazy as it is, we import stuff here to deal with config options.
    from datasets.base import get_generators
    from datasets.config import config as datasets_config
    from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
    from keras.models import load_model
    from models.base import get_model
    from models.config import config as models_config
    import time

    main(config)
