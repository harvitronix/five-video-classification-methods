from datasets.ucf101 import producer

def get_dataset(dataset, sequence_length, nb_classes, input_shapes,
                preprocessing_steps, batch_size):
    """Retrieves the appropriate dataset."""
    if dataset == 'ucf101':
        # TODO Handle multiple processors / paths. This assumes a single, plain image.
        data = producer.DataSet(sequence_length, nb_classes, input_shapes[0])
        train_gen = data.frame_generator(batch_size, 'train')
        test_gen = data.frame_generator(batch_size, 'test')
        return train_gen, test_gen
    else:
        raise ValueError("Invalid dataset.")

