import os
import random
import tensorflow as tf
import numpy as np
import math
from scipy.io import loadmat
from PIL import Image
import argparse, shutil
# from Data_Augmentation import augment
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, \
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, \
    Activation, Concatenate, Layer, Lambda
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, \
    Callback, LearningRateScheduler, TerminateOnNaN

IMAGE_SIZE = 227
ROT_RANGE = 30
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ZOOM_RANGE = 0.2
SHEAR_RANGE = 0.2
HORIZONTAL_FLIP = True
VALIDATION_BATCH_SIZE = 32


def imagenet_generator(train_data_path, val_data_path, batch_size, do_augment, val_batch_size=VALIDATION_BATCH_SIZE):
    '''
        train_data_path: Path for ImageNet Training Directory
        val_data_path: Path for ImageNet Validation Directory
        :return: Keras Data Generators for Training and Validation
    '''
    if do_augment == True:
        rot_range = ROT_RANGE
        w_shift_r = WIDTH_SHIFT_RANGE
        h_shift_r = HEIGHT_SHIFT_RANGE
        z_range = ZOOM_RANGE
        shear_r = SHEAR_RANGE
        h_flip = True
    else:
        rot_range = 0
        w_shift_r = 0.0
        h_shift_r = 0.0
        z_range = 0.0
        shear_r = 0.0
        h_flip = False

    print("Grabbing Training Dataset")
    train_datagen = ImageDataGenerator(samplewise_center=False, \
                                       rotation_range=rot_range, \
                                       width_shift_range=w_shift_r, \
                                       height_shift_range=h_shift_r, \
                                       zoom_range=z_range, \
                                       shear_range=shear_r, \
                                       horizontal_flip=h_flip, \
                                       fill_mode='nearest', rescale=1. / 255)

    val_datagen = ImageDataGenerator( rescale=1. / 255)

    '''
      Change follow_links to True when using symbolic links
      to training and validation data
    '''
    train_generator = train_datagen.flow_from_directory( \
        train_data_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), \
        batch_size=batch_size, shuffle=True, class_mode='categorical', \
        follow_links=True)
    print("Grabbing Validation Dataset")
    validation_generator = val_datagen.flow_from_directory( \
        val_data_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), \
        batch_size=val_batch_size, shuffle=True, \
        class_mode='categorical', \
        follow_links=True)
    return train_generator, validation_generator


def imagenet_generator_multi(train_data_path, val_data_path, batch_size, do_augment,val_batch_size=VALIDATION_BATCH_SIZE):
    '''
        For use with auxiliary classifiers or mutliple outputs
        train_data_path: Path for ImageNet Training Directory
        val_data_path: Path for ImageNet Validation Directory
        :return: Keras Data Generators for Training and Validation
    '''
    if do_augment == True:
        rot_range = ROT_RANGE
        w_shift_r = WIDTH_SHIFT_RANGE
        h_shift_r = HEIGHT_SHIFT_RANGE
        z_range = ZOOM_RANGE
        shear_r = SHEAR_RANGE
        h_flip = True
    else:
        rot_range = 0
        w_shift_r = 0.0
        h_shift_r = 0.0
        z_range = 0.0
        shear_r = 0.0
        h_flip = False

    print("Grabbing Training Dataset")
    train_datagen = ImageDataGenerator(samplewise_center=False, \
                                       rotation_range=rot_range, \
                                       width_shift_range=w_shift_r, \
                                       height_shift_range=h_shift_r, \
                                       zoom_range=z_range, \
                                       shear_range=shear_r, \
                                       horizontal_flip=h_flip, \
                                       fill_mode='nearest', rescale=1. / 255)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    '''
      Change follow_links to True when using symbolic links
      to training and validation data
    '''
    train_generator = train_datagen.flow_from_directory( \
        train_data_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), \
        batch_size=batch_size, shuffle=True, class_mode='categorical', \
        follow_links=True)
    print("Grabbing Validation Dataset")
    validation_generator = val_datagen.flow_from_directory( \
        val_data_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), \
        batch_size=val_batch_size, shuffle=True, \
        class_mode='categorical', \
        follow_links=True)
    multi_train_generator = create_multi_generator(train_generator)
    multi_validation_generator = create_multi_generator(validation_generator)

    train_size = train_generator.samples
    val_size = validation_generator.samples
    return multi_train_generator, multi_validation_generator, train_size, val_size


def create_multi_generator(data_generator):
    while (True):
        data_imgs, data_l = next(data_generator)
        yield [data_imgs], [data_l, data_l]


def fit_model(model, num_classes, first_class, last_class, batch_size, op_type=None, \
              decay_params=None, imagenet_path=None, \
              train_path=None, val_path=None, tb_logpath='./logs', \
              meta_path=None, config_path=None, num_epochs=1000, augment=True, \
              multi_outputs=False, clrcm_params=None,train_by_branch=False):
    '''
    :param model: Keras model
    :param num_classes:
    :param batch_size:
    :param op_type: Optimizer type
    :param decay_params: Decay parameters for rmsprop
    :param imagenet_path:
    :param train_path:
    :param val_path:
    :param tb_logpath: Tensorboard Path
    :param meta_path: ImageNet meta path
    :param config_path: Config file path
    :param num_epochs:
    :param augment: Augment data (t/f)
    :param multi_outputs: Use aux classifier
    :param clrcm_params: CLRC(Cyclical Learning Rate, Cyclical Momentum for sgd
    :return:
    '''
    '''
      Fit Model to dataset
    '''
    orig_train_img_path = os.path.join(imagenet_path, train_path)
    orig_val_img_path = os.path.join(imagenet_path, val_path)
    train_img_path = orig_train_img_path
    val_img_path = orig_val_img_path
    wnid_labels, _ = load_imagenet_meta(os.path.join(imagenet_path, \
                                                     meta_path))

    if (num_classes < 1000):
        train_img_path = os.path.join(imagenet_path, 'TrainingClasses_/')
        val_img_path = os.path.join(imagenet_path, 'ValidationClasses_/')
        create_selective_symbolic_link(first_class, last_class, wnid_labels, \
                             original_training_path=orig_train_img_path, \
                             new_training_path=train_img_path, \
                             original_validation_path=orig_val_img_path, \
                             new_validation_path=val_img_path, \
                             config_path=config_path)

    for layer in model.layers:
        print(layer, layer.trainable)
    print(model.inputs)
    print(model.outputs)
    print("Initializing Callbacks")
    tb_callback = TensorBoard(log_dir=tb_logpath)
    '''
    checkpoint_callback = ModelCheckpoint(filepath='weights.h5'\
                        ,verbose = 1, save_weights_only = True, period=1)
    '''

    termNaN_callback = TerminateOnNaN()
    save_weights_callback = SaveWeightsNumpy('weights.npy', period=5)
    callback_list = [tb_callback, save_weights_callback, termNaN_callback]

    '''
        If the training each branch individually, increase the number of epochs
        to be num_classes*num_epochs 
    '''
    if train_by_branch == True:
        each_branch_callback = TrainByBranch(num_classes,num_epochs)
        num_epochs*=num_classes

    if op_type == 'rmsprop':
        '''
            If the optimizer type is RMSprop, decay learning rate
            and append to callback list
        '''
        lr_decay_callback = ExpDecayScheduler(decay_params[0], \
                                              decay_params[1], decay_params[2])
        callback_list.append(lr_decay_callback)
    elif op_type == 'adam':
        print ('Optimizer: Adam')
    elif op_type == 'sgd':
        print('Optimizer: SGD')
        one_cycle = OneCycle(clrcm_params[0],clrcm_params[1],clrcm_params[2],\
                             clrcm_params[3], clrcm_params[4],clrcm_params[5])
        callback_list.append(one_cycle)
    else:
        print ('Invalid Optimizer. Exiting...')
        exit()
    print("Generating Data")
    # Get training and validation generators
    if multi_outputs is True:
        train_data, val_data = imagenet_generator_multi(train_img_path, \
                                                        val_img_path, batch_size=batch_size, \
                                                        do_augment=augment)
    else:
        train_data, val_data = imagenet_generator(train_img_path, val_img_path, \
                                                  batch_size=batch_size, \
                                                  do_augment=augment)
    print(train_data)
    # Fit and validate model based on generators
    print("Fitting Model")
    model.fit_generator(train_data, epochs=num_epochs, \
                        steps_per_epoch=int(num_classes*1300)/batch_size,\
                        validation_data=val_data, \
                        validation_steps= \
                            int((num_classes * 50) / VALIDATION_BATCH_SIZE), \
                        verbose=1, callbacks=callback_list)

    # save_model(model, 'google_csn.h5')

    return model


def load_model_npy(model, filename):
    print("Loading weights from: " + str(filename))
    weights = np.load(filename, encoding="latin1")

    model.set_weights(weights)
    print("WEIGHTS LOADED")
    return model

def create_selective_symbolic_link(first_class, last_class, wnid_labels, \
                         original_training_path, new_training_path, \
                         original_validation_path, new_validation_path,\
                         config_path):
    if os.path.exists(new_training_path):
      shutil.rmtree(new_training_path)
    os.makedirs(new_training_path)
    if os.path.exists(new_validation_path):
      shutil.rmtree(new_validation_path)
    os.makedirs(new_validation_path)
    class_list = wnid_labels[first_class : last_class + 1]
    for dir in class_list:
      src = os.path.join(original_training_path, dir.strip('\n'))
      dst = os.path.join(new_training_path, dir.strip('\n'))
      os.symlink(src, dst)
      src = os.path.join(original_validation_path, dir.strip('\n'))
      dst = os.path.join(new_validation_path, dir.strip('\n'))
      os.symlink(src, dst)    

def create_symbolic_link(num_classes, wnid_labels, original_training_path, \
                         new_training_path, original_validation_path, \
                         new_validation_path, config_path):
    if os.path.exists(new_training_path):
        shutil.rmtree(new_training_path)
    os.makedirs(new_training_path)

    if os.path.exists(new_validation_path):
        shutil.rmtree(new_validation_path)
    os.makedirs(new_validation_path)

    class_list = select_input_classes(num_classes, wnid_labels, \
                                      original_training_path, \
                                      original_validation_path, \
                                      config_path=config_path)

    for dir in class_list:
        src = os.path.join(original_training_path, dir.strip('\n'))
        dst = os.path.join(new_training_path, dir.strip('\n'))
        os.symlink(src, dst)
        src = os.path.join(original_validation_path, dir.strip('\n'))
        dst = os.path.join(new_validation_path, dir.strip('\n'))
        os.symlink(src, dst)


def select_input_classes(num_classes, wnid_labels, original_training_path, \
                         original_validation_path, config_path):
    class_list = []
    c_path = os.path.join(config_path, "config.txt")
    if (os.path.isfile(c_path)):
        with open(c_path, "r") as ins:
            array = []
            for line in ins:
                if "Classes" in line:
                    class_list = line.replace("Classes:", "").replace(" ", "") \
                        .split(',')
                    if len(class_list) != num_classes:
                        print("\n\nlow number of classes in config file")
                        exit()
                    if not verify_classes(class_list, original_training_path, \
                                          original_validation_path):
                        print("\n\nlow number of figures in class")
                        exit()
                    return class_list

    while len(class_list) < num_classes:
        class_index = random.randrange(1000)
        folder = wnid_labels[class_index]
        if folder not in class_list and verify_classes([folder], \
                                                       original_training_path, original_validation_path):
            class_list.append(folder)

    if os.path.exists(c_path):
        a_w = 'a'  # append if already exists
    else:
        a_w = 'w'

    file = open(c_path, a_w)
    file.write('Classes: ' + (",".join(class_list)))

    return class_list


def verify_classes(class_list, original_training_path, \
                   original_validation_path):
    if not class_list:
        return True

    for dir in class_list:
        path = os.path.join(original_training_path, dir.strip('\n'))
        if len([name for name in os.listdir(path) if \
                os.path.isfile(os.path.join(path, name))]) < 1300:
            return False

    for dir in class_list:
        path = os.path.join(original_validation_path, dir.strip('\n'))
        if len([name for name in os.listdir(path) if \
                os.path.isfile(os.path.join(path, name))]) < 50:
            return False

    return True


def onehot(index, tot_classes):
    """ It creates a one-hot vector with a 1.0 in
        position represented by index
    """
    '''new_idx = 0
    if index >= tot_classes-1:
        new_idx = tot_classes-1
    else:
        new_idx = index'''
    onehot = np.zeros(tot_classes)
    onehot[index] = 1.0
    # print(onehot)
    return onehot


def preprocess_image(image_path, augment_img=False):
    """ It reads an image, it resize it to have the lowest dimesnion of 256px,
        it randomly choose a 227x227 crop inside the resized image and normalize
        the numpy array subtracting the ImageNet training set mean
        Args:
            images_path: path of the image
        Returns:
            cropped_im_array: the numpy array of the image normalized
            [width, height, channels]
    """
    IMAGENET_MEAN = [123.68, 116.779, 103.939]  # rgb format

    img = Image.open(image_path).convert('RGB')
    # resize of the image (setting lowest dimension to 256px)
    if img.size[0] < img.size[1]:
        h = int(float(256 * img.size[1]) / img.size[0])
        img = img.resize((256, h), Image.ANTIALIAS)
    else:
        w = int(float(256 * img.size[0]) / img.size[1])
        img = img.resize((w, 256), Image.ANTIALIAS)

    x = random.randint(0, img.size[0] - 227)
    y = random.randint(0, img.size[1] - 227)
    img_cropped = img.crop((x, y, x + 227, y + 227))
    cropped_im_array = np.array(img_cropped, dtype=np.float32)

    for i in range(3):
        cropped_im_array[:, :, i] -= IMAGENET_MEAN[i]

    if augment_img == True:
        '''
            Returns original cropped image and list of cropped images

                augment 'mode' parameter:
                    0 = flip only
                    1 = flip and rotate
                    2 = flip, rotate, and translate

                augmented_imgs[0] = flipped
                augmented_imgs[1] = rotated
                augmented_imgs[2] = translated
        '''
        augmented_imgs = augment(cropped_im_array, mode=0)
        return cropped_im_array, augmented_imgs

    '''
        Otherwise just returns original cropped image
    '''
    return cropped_im_array


def load_imagenet_meta(meta_path):
    """ It reads ImageNet metadata from ILSVRC 2012 dev tool file
        Args:
            meta_path: path to ImageNet metadata file
        Returns:
            wnids: list of ImageNet wnids labels (as strings)
            words: list of words (as strings) referring to wnids labels and
            describing the classes
    """
    metadata = loadmat(meta_path, struct_as_record=False)
    '''
    ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 
    'wordnet_height', 'num_train_images']
    '''
    synsets = np.squeeze(metadata['synsets'])
    ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
    wnids = np.squeeze(np.array([s.WNID for s in synsets]))
    words = np.squeeze(np.array([s.words for s in synsets]))
    return wnids, words


def imagenet_size(im_source):
    """ It calculates the number of examples in ImageNet training-set
        Args:
            im_source: path to ILSVRC 2012 training set folder
        Returns:
            n: the number of training examples
    """
    n = 0
    for d in os.listdir(im_source):
        for f in os.listdir(os.path.join(im_source, d)):
            n += 1
    return n


'''
    Used for RMSprop optimizer for learning rate decay defined in
    Inception V4 Paper  https://arxiv.org/abs/1602.07261
'''


class ExpDecayScheduler(Callback):
    def __init__(self, initial_lr, n_epoch, decay):
        '''
        :param initial_lr: Initial Learning Rate
        :param n_epoch: Every epoch to decay learning rate
        :param decay: Decay factor
        '''
        super(ExpDecayScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.n_epoch = n_epoch
        self.decay = decay

    def on_epoch_begin(self, epoch, logs={}):
        old_lr = K.get_value(self.model.optimizer.lr)
        if epoch % self.n_epoch == 0 and epoch > 0:
            new_lr = self.initial_lr * np.exp(-self.decay * epoch)
            print("Decaying Learning Rate... New LR = " + str(new_lr))
            K.set_value(self.model.optimizer.lr, new_lr)
        else:
            K.set_value(self.model.optimizer.lr, old_lr)

class TrainByBranch(Callback):
    '''
        Train each branch on Octopus individually by setting
        (trainable=False) for all branches except desired one.
        When epoch % epoch_limit == 0 , change trainable branch.
    '''
    def __init__(self, num_classes,epoch_limit):
        '''
        :param num_classes: Number of classes
        :param epoch_limit: Epoch Limit to train each branch
        '''
        super(TrainByBranch, self).__init__()
        self.num_classes = num_classes
        self.epoch_limit = epoch_limit
        self.base_name_arr = ['inception_4c/1x1',\
                         'inception_4c_3x3_reduce',\
                         'inception_4c/5x5_reduce',\
                         'inception_4c/3x3',\
                         'inception_4c/5x5',\
                         'inception_4c/pool_proj',\
                         'inception_4d_1x1',\
                         'inception_4d_3x3_reduce',\
                         'inception_4d_5x5_reduce', \
                         'inception_4d_3x3', \
                         'inception_4d_5x5', \
                         'inception_4d_pool_proj', \
                         'inception_4e_1x1', \
                         'inception_4e_3x3_reduce', \
                         'inception_4e_5x5_reduce', \
                         'inception_4e_3x3', \
                         'inception_4e_5x5', \
                         'inception_4e_pool_proj', \
                         'inception_5a_1x1', \
                         'inception_5a_3x3_reduce', \
                         'inception_5a_5x5_reduce', \
                         'inception_5a_3x3', \
                         'inception_5a_5x5', \
                         'inception_5a_pool_proj', \
                         'inception_5b_1x1', \
                         'inception_5b_3x3_reduce', \
                         'inception_5b_5x5_reduce', \
                         'inception_5b_3x3', \
                         'inception_5b_5x5', \
                         'inception_5b_pool_proj',\
                         '_loss3/classifier']
        self.branch_num = 0

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.epoch_limit == 0:
            print("\n\nTRAINING BRANCH "+ str(self.branch_num)+"\n\n")
            for n in self.base_name_arr:
                #Set current branch to true
                K.set_value(self.model.get_layer(str(self.branch_num)+n).trainable,True)
                #Set previous branch to false (if not first branch)
                if self.branch_num>0:
                    K.set_value(self.model.get_layer(str(self.branch_num-1) + n).trainable, False)

            self.branch_num+=1
            #Iterate through layers to double check 'trainable'
            #Comment out when debugged
            for layer in model.layers:
                print(layer, layer.trainable)


class OneCycle(Callback):
    def __init__(self, min_lr, max_lr, min_mom, max_mom, step_size, div):
        '''
        As defined by Smith in arXiv:1803.09820
        
        :param min_lr: Minimum Learning Rate
        :param max_lr: Maximum Learning Rate
        :param min_mom: Minimum Momentum
        :param max_mom: Maximum Momentum
        :param step_size: number of iterations per half-cycle
        :param div: 
        '''
        super(OneCycle, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_mom = min_mom
        self.max_mom = max_mom
        self.step_size = step_size
        self.div = div

    def on_batch_begin(self, batch, logs={}):
        iteration = float(K.get_value(self.model.optimizer.iterations))
        cycle = math.floor(1.0 + iteration / (2.0 * self.step_size))
        x = math.fabs(iteration/self.step_size - 2.0 * cycle + 1.0)
        print("\nIteration: " + str(iteration))
        print("Cycle: " + str(cycle))
        print("X: "+str(x)) 
        if iteration > 2*self.step_size:
            #Set learning rate depending on iteration
            new_lr = self.min_lr - (self.min_lr - (self.min_lr/self.div)) * \
                     math.fabs(1.0 - x)
            #allow lr to decay further to several order of mag lower than min
            if iteration>(2+1)*self.step_size:
                new_lr = (self.min_lr/self.div)
            K.set_value(self.model.optimizer.lr, new_lr)
            #Set momentum to max after cycle
            new_mom = self.max_mom
            K.set_value(self.model.optimizer.momentum, new_mom)
            print("Setting learning rate: " + str(new_lr))
            print("Setting momentum: " + str(new_mom))
        else:
            print("Min_lr: " + str(self.min_lr))
            print("Max_lr: " + str(self.max_lr))
            print("Max_lr - Min_lr: " + str((self.max_lr - self.min_lr)))
            print("abs(1.0-x) : " + str(math.fabs(1.0 - x)))
            #Set learning rate depending on iteration
            new_lr = self.min_lr + (self.max_lr - self.min_lr) * math.fabs(1.0 - x)
            K.set_value(self.model.optimizer.lr, new_lr)
            #Set momentum depending on iteration
            new_mom = self.max_mom - (self.max_mom - self.min_mom) * \
                      math.fabs(1.0 - x)
            K.set_value(self.model.optimizer.momentum, new_mom)
            print("Setting learning rate: " + str(new_lr))
            print("Setting momentum: " + str(new_mom))
        
        


class SaveWeightsNumpy(Callback):
    def __init__(self, file_path, period):
        super(SaveWeightsNumpy, self).__init__()
        self.file_path = file_path
        self.period = period

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0 and epoch >= 5:
            weights = self.model.get_weights()
            print("Saving weights to: " + str(self.file_path))
            np.save(self.file_path, weights)
