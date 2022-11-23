import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing import image
from keras.utils import load_img, img_to_array,array_to_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization
from eye_track import *
import CNN
# data 가져오기
open_list = [f for f in os.listdir('./Open/') if not f.startswith('.')]
close_list = [f for f in os.listdir('./Closed/') if not f.startswith('.')]

# 눈 이미지들이 약 80x80 ~ 110x110이여서 평균 100x100으로 잡음
img_w, img_h = 100,100

images = []  # 실제 데이터
labels = []  # 정답 데이터(1,0으로 분류)

for i in close_list:
    image = load_img('./Closed/' + i, target_size=(img_w, img_h))
    image = img_to_array(image)  # 이미지를 수로 이루어진 array로 만듦
    images.append(image)
    labels.append(1)  # 눈 감았을 때 : 1
for i in open_list:
    image = load_img('./Open/' + i, target_size=(img_w, img_h))
    image = img_to_array(image)
    images.append(image)
    labels.append(0)  # 눈 떴을 때 : 0

#전체 data 중 훈련 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

'''
저장시킨 모델 있나 없나 확인
if isfile == True인 경우 저장한 모델 가져옴
else 모델 학습
'''
model_file = './eye_model.h5'
if os.path.isfile(model_file):
    model = load_model('eye_model.h5')
else:
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(img_w, img_h, 3), activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPool2D(pool_size=2))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='softmax'))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=10)
    # model.save('eye_model.h5')

    model = Sequential()
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

    # prior layer should be flattend to be connected to dense layers
    model.add(Flatten())
    # dense layer with 50 neurons
    model.add(Dense(50, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    # final layer with 10 neurons to classify the instances
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=10)
    model.save('eye_model.h5')

preditions = []
for i in videos: #video들이 담겨 있는 arr
    prediction = np.argmax(model.predict(i),axis=-1)
    preditions.append(prediction)


#Judge_drawsy ---------------------

# 5초 x 초당 10개의 눈 사진 -> 요게 n개가 들어온다. 
images = [] # (눈동자 이미지 50개 배열 / 영상) 들의 배열 
ans=[] #(정답 label / 영상)

#CNN모델 instance 생성
#using output from main.py

predicted = [] #예측한 눈동자 상태 배열이 안에 append 된다.
mymodel = CNN()
for i in len(ans):
    predicted.append(mymodel.prediction(images[i]))


#train 및 test데이터 분리 
x_train, x_test, y_train, y_test = train_test_split(np.array(predicted), np.array(ans), test_size=0.2)


model_file = './is_sleeping.h5'
if os.path.isfile(model_file):
    model = load_model('eye_model.h5')
else:
    #binary classification 
    model = Sequential()
    model.add(Dense(150, activation='relu', input_shape=(50,))) # 10개의 눈에 대한 close or open / second  1차원 row
    model.add(Dropout(0.5))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    #손실함수 binary_crossentropy 
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=5)
    model.save('is_sleeping.h5')


#test 데이터 돌려서 plotting 
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
test_prediction = np.argmax(model.predict(x_test), axis=-1)

#plt로 accuracy 표시 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 훈련 과정 시각화 (손실)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()











