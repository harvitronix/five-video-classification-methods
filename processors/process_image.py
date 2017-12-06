import numpy as np
from keras.preprocessing.image import img_to_array, load_img

def process_image(image, target_shape):
    """Return a normalized image numpy array.
    
    image input can be numpy array or path to image. If path given, we'll
    load it first.
    """
    if isinstance(image, str):
        h, w, _ = target_shape
        image_data = load_img(image_path, target_size=(h, w))
        image_arr = img_to_array(image_data)
    else:
        image_arr = image

    # Normalize and return.
    return (image_arr / 255.).astype(np.float32)
