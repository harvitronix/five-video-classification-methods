"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import os.path
import sys, json

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100, repoDir='', featureFilePath='data/data_file.csv', workDir = 'data', lr=1e-5, decay=1e-6, classlist=[]):

         
    # Helper: Save the model.
    checkpointpath = os.path.join(workDir, 'checkpoints')
    if not os.path.exists(checkpointpath):
        print ("Creating checkpoint folder [%s]",  checkpointpath)
        os.makedirs(checkpointpath)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(workDir, 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    logpath = os.path.join(workDir, 'logs')
    if not os.path.exists(logpath):
        print ("Creating log folder [%s]",  logpath)
        os.makedirs(logpath)    
    tb = TensorBoard(log_dir=os.path.join(workDir, 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(logpath, model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            repoDir = repoDir,
            featureFilePath = featureFilePath,
            workDir=workDir,
            classlist = classlist
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape,
            repoDir = repoDir,
            featureFilePath = featureFilePath,
            workDir=workDir,
            classlist = classlist
        )
    # Check if data is sufficient
    if False == data.check_data(batch_size):
        sys.exit(0)

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model, lr, decay)

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
    batch_size = 32
    nb_epoch = 1000
    featureFilePath='data/data_file.csv'
    lr = 1e-5
    decay = 1e-6    

    #read config file
    if len(sys.argv) > 1:
        configfile = sys.argv[1]
    else:
        print ("Usage: script <fullpath to config.json>")
        sys.exit(0)
    with open(configfile, "r") as config_file:
        config = json.load(config_file)
    featureFilePath = os.path.join(config['globaldataRepo'], config["featurefile"])
    classlist = config["eventtypes"]
    if not os.path.exists(featureFilePath):
       print ("event csv path ", featureFilePath, " doesn't exist, exiting")
       sys.exit(0)
    workDir = os.path.join(config['globaldataRepo'], config['sessionname'])
    repoDir = config['globaldataRepo']
    for videoConfig in config['training']:
        if videoConfig["modality"] == "video":
            model = videoConfig["algorithm"]
            saved_model = None  # None or weights file
            seq_length = videoConfig["sequencelength"]
            if  videoConfig["loadtomemory"] == 1:
                load_to_memory = True  # pre-load the sequences into memory
            batch_size = videoConfig["batchsize"]
            nb_epoch = videoConfig["epochs"]      
            lr = videoConfig['learningrate']
            decay = videoConfig['decay']
       
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
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch,
          repoDir = repoDir,
          featureFilePath=featureFilePath,
          workDir=workDir,
          lr = lr,
          decay = decay,
          classlist = classlist)

if __name__ == '__main__':
    main()
