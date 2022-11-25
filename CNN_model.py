import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing import image
from keras.utils import load_img, img_to_array,array_to_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization
import pickle

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
    if os.path.isfile("eye_model3.h5"):
        model = load_model("eye_model3.h5")
        history = pickle.load(open('trainHistoryDict', "rb"))
    else:
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

        model.save('eye_model3.h5')
        with open('trainHistoryDict', 'wb') as file_pi:
            pickle.dump(model.history, file_pi)

      

    return model,history

def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['training', 'validation'], loc = 'upper left')
    plt.show()

# 데이터 가져오기
x_train, x_test, y_train, y_test = load_data()

# model 학습 
epochs = 50
batch_sizes = 10
model, history = make_model(x_train, x_test, y_train, y_test,epochs,batch_sizes)

# Accuracy plot
plot_history(history)

