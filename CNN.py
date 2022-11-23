import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing import image
from keras.utils import load_img, img_to_array,array_to_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D,MaxPool2D,Flatten,BatchNormalization


class CNN:
    def __init__(self):
        # data 가져오기
        open_list = [f for f in os.listdir('./Open/') if not f.startswith('.')]
        close_list = [f for f in os.listdir('./Closed/') if not f.startswith('.')]

        # 눈 이미지들이 약 80x80 ~ 110x110이여서 평균 100x100으로 잡음
        self.img_w, self.img_h = 100,100

        images = []  # 실제 데이터
        labels = []  # 정답 데이터(1,0으로 분류)

        for i in close_list:
            image = load_img('./Closed/' + i, target_size=(self.img_w, self.img_h))
            image = img_to_array(image)  # 이미지를 수로 이루어진 array로 만듦
            images.append(image)
            labels.append(1)  # 눈 감았을 때 : 1
        for i in open_list:
            image = load_img('./Open/' + i, target_size=(self.img_w, self.img_h))
            image = img_to_array(image)
            images.append(image)
            labels.append(0)  # 눈 떴을 때 : 0
        
        self.images = images
        self.labels = labels
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.array(images), np.array(labels), test_size=0.2)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.1)
        self.model = None

    def make_a_model(self):
        model_file = './eye_model.h5'
        if os.path.isfile(model_file):
            model = load_model('eye_model.h5')
        else:
            model = Sequential()

            model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(self.img_w, self.img_h, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=2)) 

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=2)) #pooling(max값을 뽑아줌)

            model.add(Flatten()) #평탄화(1차원 vector) 
            model.add(Dense(128, activation='relu')) #128개의 RElu activation function
            model.add(Dense(2, activation='softmax')) #높은 확률을 softmax로 뽑아냄-> 0 or 1

            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model = model
    
    def train(self):
        history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=30, batch_size=5)
        self.model.save('eye_model.h5')
        return history

    def prediction(self,image_arr): #image배열을 받아 0 또는 1로 이루어진 list return 
        test_prediction = np.argmax(self.model.predict(image_arr), axis=-1)
        return test_prediction

    

    


