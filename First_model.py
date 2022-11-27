import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization

def load_data():
    # data 가져오기
    open_list = [f for f in os.listdir('./open_eye/') if not f.startswith('.')]
    close_list = [f for f in os.listdir('./close_eye/') if not f.startswith('.')]

    images = []  # 실제 데이터
    labels = []  # 정답 데이터(1,0으로 분류)

    # 눈 이미지들이 약 80x80 ~ 110x110이여서 평균 100x100으로 잡음
    img_w, img_h = 100,100
    for i in close_list:
        image = load_img('./close_eye/' + i, target_size=(img_w, img_h))
        image = img_to_array(image)  # 이미지를 수로 이루어진 array로 만듦
        images.append(image)
        labels.append(1)  # 눈 감았을 때 : 1
    for i in open_list:
        image = load_img('./open_eye/' + i, target_size=(img_w, img_h))
        image = img_to_array(image)
        images.append(image)
        labels.append(0)  # 눈 떴을 때 : 0

    #전체 data 중 훈련 데이터 나누기
    x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    return x_train, x_test, y_train, y_test

def make_model(x_train, x_test, y_train, y_test,epochs,batch_sizes):
    model = Sequential()
    # 눈 이미지들이 약 80x80 ~ 110x110이여서 평균 100x100으로 잡음
    img_w, img_h = 100,100
    model.add(Conv2D(input_shape = (img_w, img_h, 3), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', kernel_initializer='he_normal',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))

    model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_sizes)

    model.save('fitted_first_model.h5')

    return model,history

def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['training', 'validation'], loc = 'upper left')
    plt.show()

def blink_prediction(crop_images):
    if os.path.isfile("fitted_first_model.h5"): # 모델 파일이 있는 경우
        model = load_model("fitted_first_model.h5")
        images_prediction = []
        images_prediction.append(np.argmax(model.predict(crop_images), axis=-1))
        images_prediction = np.array(images_prediction)
        return images_prediction
    else: # 만들어둔 모델 파일이 없는 경우
        x_train, x_test, y_train, y_test = load_data()
        epochs = 100
        batch_sizes = 10
        model,history = make_model(x_train, x_test, y_train, y_test,epochs,batch_sizes)
        images_prediction = []
        images_prediction.append(np.argmax(model.predict(crop_images), axis=-1))
        images_prediction = np.array(images_prediction)
        return images_prediction

    
