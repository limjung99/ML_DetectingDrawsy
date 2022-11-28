import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# 눈 각도 돌려가면서 데이터셋 추가
def load_data_gen(x_train,y_train):
    datagen = ImageDataGenerator(rotation_range = 90,
                             width_shift_range=0.4,
                             height_shift_range=0.4,
                             vertical_flip =True,
                             horizontal_flip =True)
    gen_x_train = []
    gen_y_train = []
    for x in range(0,len(x_train)):
        idx = 0
        t = x_train[x].reshape((1,) + x_train[x].shape)
        for batch in datagen.flow(t , batch_size=1): # 여기서 batch는 t가 됨
            gen_x_train.append(batch[0])
            gen_y_train.append(y_train[x])
            idx += 1
            if idx%6 == 0:
                break
    
    return np.array(gen_x_train),np.array(gen_y_train)

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
    x_train,y_train = load_data_gen(x_train,y_train)
    return x_train, x_test, y_train, y_test

def make_model(x_train, x_test, y_train, y_test,epochs,batch_sizes):
    model = Sequential()
    # 눈 이미지들이 약 80x80 ~ 110x110이여서 평균 100x100으로 잡음
    img_w, img_h = 100,100
    model.add(Conv2D(32, (3,3), padding = 'same', input_shape = (img_w,img_h,3), activation = 'relu'))
    model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128,(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256,(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))

    model.summary()
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_sizes)

    # model.save('fitted_first_model.h5')

    return model,history

# history 도표로 나타내는 함수
def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['training', 'validation'], loc = 'upper left')
    plt.show()

# crop한 이미지를 가져와 CNN 모델에 적용하여 prediction값을 반환한다.
def blink_prediction(crop_images):
    if os.path.isfile("fitted_first_model.h5"): # 모델 파일이 있는 경우
        model = load_model("fitted_first_model.h5")
        print(model.summary())
        images_prediction = []
        images_prediction.append(np.argmax(model.predict(crop_images), axis=-1))
        images_prediction = np.array(images_prediction)
        print("------------1차모델 predict 결과------------")
        print(images_prediction)
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
