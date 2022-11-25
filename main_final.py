import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout
from eye_track import *
from CNN_model import * 
import os




# 5초 x 초당 10개의 눈 사진 -> 요게 n개가 들어온다. 
# load data
with open('videos_array', 'rb') as fr:
    videos_arr = pickle.load(fr)

with open("videos_label","rb") as f:
    ans = pickle.load(f)




predicted = [] #예측한 눈동자 상태 배열이 안에 append 된다.
mymodel = model #CNN모델 import 

for i in videos_arr:
    predicted.append(np.argsort(mymodel.prediction(videos_arr[i]),axis=-1))




#train 및 test데이터 분리 
x_train, x_test, y_train, y_test = train_test_split(np.array(predicted), np.array(ans), test_size=0.2)


model_file = './is_sleeping.h5'
if os.path.isfile(model_file):
    model = load_model('is_sleeping.h5')
else:
    #binary classification 
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(50,))) # 10개의 눈에 대한 close or open / second 
    model.add(Dropout(0.5))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    #손실함수 binary_crossentropy 
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=5)
    model.save('is_sleeping.h5')


#test 데이터 돌려서 plotting 
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
test_prediction = np.argmax(model.predict(x_test), axis=-1)







'''
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
'''





