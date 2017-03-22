"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data import DataSet

def validate(data_type, model, seq_length=40, saved_model=None,
             concat=False, class_limit=None, image_shape=None):
    batch_size = 32

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

    val_generator = data.frame_generator(batch_size, 'test', data_type, concat)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Evaluate!
    results = rm.model.evaluate_generator(
        generator=val_generator,
        val_samples=3200)

    print(results)
    print(rm.model.metrics_names)

def main():
    model = 'mlp'
    saved_model = 'data/ucf101/checkpoints/mlp-features.023-0.926.hdf5'

    if model == 'conv_3d' or model == 'crnn':
        data_type = 'images'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None

    if model == 'mlp':
        concat = True
    else:
        concat = False

    validate(data_type, model, saved_model=saved_model,
             concat=concat, image_shape=image_shape)

if __name__ == '__main__':
    main()
