from datasets.ucf101 import producer as ucf_producer
from datasets.synthetic_boxes import producer as synthetic_producer
from processors.process_image import process_image
import threading
import numpy as np

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

def preprocess(data, input_shape, process):
    # run the process on the data.
    if process == 'images':
        processed_data = np.array(
            [[process_image(x, input_shape) for x in row] for row in data]
        )
    else:
        raise ValueError("Unknown preprocess type: %s" % process)

    return processed_data

@threadsafe_generator
def generate_batch(preprocessing_steps, input_shapes, producer_gen):
    # get a batch from the producer
    for x, y in producer_gen:

        # process it across all preprocessors
        x_paths = [preprocess(x, input_shape, process)
                   for input_shape, process 
                   in zip(input_shapes, preprocessing_steps)]

        # return the array of arrays of data to yield
        yield x_paths, y

def get_generators(dataset, sequence_length, nb_classes, input_shapes,
                   preprocessing_steps, batch_size, params=None):
    """
    Retrieves the appropriate dataset, runs it through appropriate processors,
    and provides a list of generators of its own.

    The dataset producers should just produce sequences of paths, rather than actually
    processed images. That way we can process as needed here.

    TODO: Explain this further, once I decide how best to do it.
    TODO: Let us pass arbitrary parameters through here into the producers
    """
    if dataset == 'ucf101':
        data = ucf_producer.DataSet(sequence_length, nb_classes)
        train_gen = data.frame_generator(batch_size, 'train')
        test_gen = data.frame_generator(batch_size, 'test')
    elif dataset == 'synthetic_boxes':
        # TODO Assumes the first (or only) path is an image shape
        dataset = synthetic_producer.SyntheticBoxes(
            batch_size, input_shapes[0][0], input_shapes[0][1],
            nb_frames=sequence_length)
        # Train and test are the same since it's all synthetically-generated data.
        train_gen = test_gen = dataset.data_gen()
    else:
        raise NotImplemented("Invalid dataset.")

    processed_train_gen = generate_batch(preprocessing_steps, input_shapes, train_gen)
    processed_test_gen = generate_batch(preprocessing_steps, input_shapes, test_gen)

    return processed_train_gen, processed_test_gen
