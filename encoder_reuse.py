import model as model
import keras.layers as layers
import keras.models as models
import keras.optimizers as opt
import keras.losses as lss
import numpy as np
import matplotlib.pyplot as plt


def get_conv_deco():
    l0 = layers.Input((10,), name ='input')
    l2 = layers.Dense(100,activation='relu')(l0)
    l3 = layers.Reshape((10,10,1), name= 'reshaping')(l2)
    l4 = layers.Conv2D(128,(3,3), activation='relu', name = 'firconv')(l3)
    l5 = layers.UpSampling2D(name= 'firstUpsample')(l4) #16

    l6 = layers.Conv2D( 64, (3,3), activation= 'relu',name='sec_conv')(l5)
    #l7 = layers.Conv2D( 64, (3,3), activation= 'relu',name='tred_conv')(l6)
    l8 = layers.UpSampling2D(name = 'upsampling2')(l6)
    l9 = layers.Conv2D(1,(1,1), activation='relu')(l8)
    return models.Model(l0,l9)

def resumple_useing_old_model(input_shape= (28,28,1)):
    vae, enco, deco = model.load_saved_model(input_shape)
    new_deco = get_conv_deco();
    enco.trainable= False
    modl =  models.Sequential([enco, new_deco])
    modl.compile(opt.adam(.01), lss.mean_squared_logarithmic_error, ['acc'])
    return modl


if __name__ == '__main__' :
    conv_deco = resumple_useing_old_model()
    conv_deco.summary()
    conv_deco.layers[1].summary()


