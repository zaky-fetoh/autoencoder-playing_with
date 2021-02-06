import numpy as np
import keras.datasets as datasets


def get_MNIST():
    (X_train, Y_train),(X_test, Y_test) = datasets.mnist.load_data()
    image_set = np.concatenate([X_train, X_test], axis= 0 )
    image_set = np.reshape(image_set,image_set.shape + (1,) )
    return (image_set - image_set.min() )/ ( image_set.max() - image_set.min())

