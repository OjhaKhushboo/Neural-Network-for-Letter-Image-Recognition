from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import regularizers
from keras import initializers
import matplotlib.pyplot as plt
from keras.models import load_model

# Path of the image files
path_im = "/Users/swaraj/desktop/LetterImage"

# Data generators and generation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

# Hyper-parameters
epochs = 5
batch = 500
train_data = 26416
valid_data = 5174

train = train_datagen.flow_from_directory(
        path_im+'/Train',
        target_size=(128, 128),
        batch_size=batch,
        class_mode='categorical')

valid = test_datagen.flow_from_directory(
        path_im+'/Validation',
        target_size=(128, 128),
        batch_size=batch,
        class_mode='categorical')

test = test_datagen.flow_from_directory(
        path_im+'/Test',
        target_size=(128, 128),
        batch_size=batch,
        class_mode='categorical')

# CNN model: 2x[Convolution layer + Pooling layer] + Flatten layer + 2 Fully-Connected layers
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dense(26, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Start CNN training
mod = model.fit_generator(
        train,
        steps_per_epoch=train_data // batch,
        epochs=epochs,
        validation_data=valid,
        validation_steps=valid_data // batch)

model.save_weights('50_epochs.h5')

# Simple Neural Network
init = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

norm = Sequential()
norm.add(Dense(20, input_shape=(128, 128, 3), kernel_initializer=init, bias_initializer=init, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
norm.add(Flatten())
norm.add(Dense(10, kernel_initializer=init, bias_initializer=init, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
norm.add(Dense(26, kernel_initializer=init, bias_initializer=init, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

norm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Start ANN training
nor = norm.fit_generator(
        train,
        steps_per_epoch=train_data // batch,
        epochs=epochs,
        validation_data=valid,
        validation_steps=valid_data // batch)

norm.save_weights('50_epochs_nor.h5')

# Plots

# Comparing accuracy of cnn and ann
plt.subplot(211)
plt.plot(mod.history['acc'])
plt.plot(nor.history['acc'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['CNN', 'Simple NN'], loc='upper left')

plt.subplot(212)
plt.plot(mod.history['val_acc'])
plt.plot(nor.history['val_acc'])
plt.title('Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['CNN', 'Simple NN'], loc='upper left')

plt.show()

# Comparing loss of cnn and ann
plt.subplot(211)
plt.plot(mod.history['loss'])
plt.plot(nor.history['loss'])
plt.title('Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['CNN', 'Simple NN'], loc='upper left')

plt.subplot(212)
plt.plot(mod.history['val_loss'])
plt.plot(nor.history['val_loss'])
plt.title('Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['CNN', 'Simple NN'], loc='upper left')

plt.show()

# CNN model testing
model.load_weights('50_epochs.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn = model.evaluate_generator(test, workers=2)
cnn_loss = cnn[0]
cnn_acc = cnn[1]
print("CNN Test evaluation\t Loss = {0:.2f}, Accuracy = {1:.01%}".format(cnn_loss, cnn_acc))


# ANN model testing
norm.load_weights('50_epochs_nor.h5')
norm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ann = norm.evaluate_generator(test, workers=2)
ann_loss = ann[0]
ann_acc = ann[1]
print("ANN Test evaluation\t Loss = {0:.2f}, Accuracy = {1:.01%}".format(ann_loss, ann_acc))

