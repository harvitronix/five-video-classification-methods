"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import os.path

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100, n_gpus=8, cnn_model_type='nasnet'):
    # Helper: Save the model.
    if cnn_model_type == 'InceptionV3':
        cnn_feature_size = 2048
    elif cnn_model_type == 'nasnet':
        cnn_feature_size = 4032
    else:
        raise(IOError('invalid cnn_model_type {}'.format(cnn_model_type)))

    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model + time.strftime("%Y%m%d_%H%M%S")))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type,cnn_model_type=cnn_model_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type,cnn_model_type=cnn_model_type)
    else:
        # Get generators.
        generator     = data.frame_generator(batch_size, 'train', data_type,cnn_model_type=cnn_model_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type,cnn_model_type=cnn_model_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model_type=model, seq_length = seq_length,
                        saved_model = saved_model, n_gpus=n_gpus, cnn_feature_size= cnn_feature_size)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)

def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lstm'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None

    seq_length = 40
    load_to_memory = False  # pre-load the sequences into memory
    n_gpus = 8
    batch_size = 32* n_gpus
    nb_epoch = 1000
    cnn_model_type = 'nasnet'



    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch,n_gpus=n_gpus,cnn_model_type=cnn_model_type)

if __name__ == '__main__':
    main()
