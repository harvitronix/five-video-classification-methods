"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os.path
import sys
from data import DataSet
from extractor import Extractor
from tqdm import tqdm
from yottato.yottato import yottato as yto

#read config file
if len(sys.argv) > 1:
    configfile = sys.argv[1]
else:
    print ("Usage: script <fullpath to config.json>")
    sys.exit(0)
yto_config = yto(sys.argv[1])

# Set defaults.
seq_length = yto_config.videoSeqLength 
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.

# Get the dataset.
data = DataSet(seq_length=seq_length,
            class_limit=class_limit,
            repo_dir = yto_config.repoDir,
            feature_file_path = yto_config.featureFileName,
            work_dir=yto_config.workDir)

# get the model.
model = Extractor()

# Loop through data.
pbar = tqdm(total=len(data.data))
sequence_path = os.path.join(yto_config.workDir, 'sequences')
if not os.path.exists(sequence_path):
    print ("Creating sequence folder [%s]",  sequence_path)
    os.makedirs(sequence_path)
for video in data.data:

    # Get the path to the sequence for this video.
    path = os.path.join(sequence_path, video[2] + '-' + str(seq_length) + \
        '-features')  # numpy will auto-append .npy

    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(yto_config.repoDir, video)

    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    # Save the sequence.
    np.save(path, sequence)

    pbar.update(1)

pbar.close()
