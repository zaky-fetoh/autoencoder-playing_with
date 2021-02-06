import utility
import encoder_reuse as enre
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
image_set = utility.get_MNIST()
conv_model = enre.resumple_useing_old_model()

history = conv_model.fit(image_set, image_set,
                         batch_size= 128, epochs=10)

rand = np.random.randint(0, image_set.shape[0], (4))

for i in range( 4 ):
    realim = image_set[rand[i]]; predim= conv_model.predict(realim[np.newaxis,...])
    realim, predim = [np.reshape(x,(28,28)) for x in [realim, predim]]
    plt.subplot(4,2, i*2 + 1); plt.imshow( realim, cmap= 'gray')
    plt.subplot(4,2, i*2 + 2); plt.imshow( predim, cmap= 'gray')

n= 4
randSamples = K.eval(K.random_normal((n*n,10)))
predic_imag = conv_model.layers[1].predict(randSamples)
for i in range(n*n):
    plt.subplot(n,n,i+1); plt.axis('off')
    plt.imshow(np.reshape(predic_imag[i,:],(28,28)))
