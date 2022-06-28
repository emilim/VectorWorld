from tensorflow import keras
from keras import layers

def CAE(input_shape=(360, 640, 3), filters=[32, 64, 128, 10]):
    model = keras.Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(layers.Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(layers.Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(layers.Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=filters[3], name='embedding'))
    model.add(layers.Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(layers.Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(layers.Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(layers.Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(layers.Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model