from tensorflow import keras


def convolutional_model(input_shape):
    # Define a convolutional model with two convolutional layers and two max pooling layers
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)

    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)

    #reshape to RNN input
    #shape = keras.backend.int_shape(x)
    #x = keras.layers.Reshape(target_shape=(shape[1], shape[2] * shape[3]))(x)

    #x = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(x)
    #x = keras.layers.Bidirectional(keras.layers.LSTM(16))(x)

    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x)
    #x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)                                                                                                                                                                                                                        
    x = keras.layers.Dense(5, activation='softmax')(x)

    conv_model = keras.Model(inputs=inputs, outputs=x)
  
    conv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return conv_model


def predict_cnn(model, x_test):
    y_predict = model.predict(x_test)
    y_predict = y_predict.argmax(axis=1)
    return y_predict