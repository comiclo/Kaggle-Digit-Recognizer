from preprocessing import load_data
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, AveragePooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint

train_data, train_labels, test_data = load_data()

batch_size = 128
nb_classes = 10
nb_epoch = 20

model = Sequential()
model.add(Conv2D(32, 5, 5, input_shape=(28, 28, 1),
                 border_mode='same', activation='relu'))
model.add(AveragePooling2D(border_mode='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, 5, 5, border_mode='same', activation='relu'))
model.add(AveragePooling2D(border_mode='same'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              # optimizer=optimizers.Adagrad(lr=1e-4),
              optimizer=optimizers.adam(lr=1e-4),
              metrics=['accuracy'])

filepath = 'mnist.model'
checkpoint= ModelCheckpoint(filepath, save_best_only=True)

history = model.fit(train_data, train_labels,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1,
                    # validation_split=0.1,
                    callbacks=[checkpoint]
                    )

model.save(filepath)



print(history.history.keys())