import keras.models as models
import keras.layers as layers
import keras.optimizers as opt
import keras.losses as lss
import keras.backend as K
import numpy as np

"""
Dense Auto_encoder
"""


def get_encoder(input_shape):
    def sampling(tensor):
        mean, vari, x = tensor
        nor = (x- mean) / (vari+ .0001)
        #gauss = lambda mie, sig, x: K.exp(-K.pow((x - mie) / sig, 2.) / 2.) / sig * np.sqrt(2 * np.pi)
        #return gauss(mean, vari, x)
        return nor

    l0 = layers.Input(input_shape, )
    l1 = layers.Flatten()(l0)
    l2 = layers.Dense(128, activation='relu')(l1)
    l3_mean = layers.Dense(10, activation='relu')(l2)
    l3_var = layers.Dense(10, activation='relu')(l2)

    cl1 = layers.Conv2D(32, (3, 3), activation='relu', name='fir_conv')(l0)
    pl1 = layers.MaxPool2D(name='firstPooling')(cl1)
    cl2 = layers.Conv2D(64, (3, 3), activation='relu', name='sec_conv')(pl1)
    pl2 = layers.MaxPool2D(name='sec_pooling')(cl2)
    lfft = layers.Flatten(name='fltting_conv_layer')(pl2)
    cl3 = layers.Dense(10, activation='relu')(lfft)

    l4 = layers.Lambda(sampling, output_shape=(10,),
                       name='final_com')([l3_mean, l3_var, cl3])

    return models.Model(l0, l4)


def get_decoder(out_shape):
    l0 = layers.Input((10,), name='decoder_input')
    l1 = layers.Dense(128, activation='relu', name='decoderHL')(l0)
    l2 = layers.Dense(np.prod(out_shape), activation='sigmoid', name='out')(l1)
    l3 = layers.Reshape(out_shape)(l2)
    return models.Model(l0, l3)


def get_vae(input_shape):
    enco = get_encoder(input_shape)
    deco = get_decoder(input_shape)
    model = models.Sequential([enco, deco])
    model.compile(opt.adam(.01), lss.mean_squared_logarithmic_error, ['acc'])
    return model

def load_saved_model( input_shape, file_name= 'vae.model.h5'):
    model = get_vae(input_shape)
    model.load_weights(file_name)
    enco, deco= model.layers
    return model, enco, deco

if __name__ == '__main__' :
    m = get_encoder((28, 28, 1))
    m.summary()
    print(m.predict_on_batch(np.random.rand(1, 28, 28, 1)))