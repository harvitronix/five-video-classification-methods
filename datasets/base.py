from datasets.ucf101 import producer

def get_dataset(dataset, sequence_length, nb_classes, input_shapes,
                preprocessing_steps, batch_size):
    """Retrieves the appropriate dataset."""
    if dataset == 'ucf101':
        data = producer.DataSet(sequence_length, nb_classes, input_shape[0])
        train_gen = data.frame_generator(batch_size, 'train')
        test_gen = data.frame_generator(batch_size, 'test')
        return train_gen, test_gen
    else:
        raise ValueError("Invalid dataset.")

