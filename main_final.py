import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout
from CNN_model import * 
import os

seed =5
np.random.seed(seed)
kfold = KFold(n_splits=2,shuffle=True,random_state=seed)

# (50개의 눈 사진 ) 들의 배열  
# load data
with open('videos_array2', 'rb') as fr:
    videos_arr = pickle.load(fr)
fr.close()
with open("video_labels2","rb") as f:
    ans = pickle.load(f)
f.close()

print(np.array(videos_arr).shape)
print(np.array(ans).shape)


# with open("videos_array2","wb") as va:
#     pickle.dump(vid_arr,va)
#     va.close()
# with open("video_labels2","wb") as vl:
#     pickle.dump(vid_label, vl)
#     vl.close()

predicted = [] #예측한 눈동자 상태 배열이 안에 append 된다.
mymodel = model #CNN모델 import 


if os.path.isfile("predicted"):
    with open("predicted","rb") as f:
        predicted = pickle.load(f)
    f.close()
else:
    for i in videos_arr:
        predicted.append(np.argmax(mymodel.predict(i),axis=-1))
    with open("predicted","wb") as f:
        pickle.dump(predicted,f)
    f.close()




# #train 및 test데이터 분리 
x_train, x_test, y_train, y_test = train_test_split(np.array(predicted), np.array(ans), test_size=0.2)
# validation 데이터 분리 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)


model_file = './is_sleeping.h5'

# if os.path.isfile(model_file):
#     model = load_model('is_sleeping.h5')
# else:
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(50,))) 
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.3))
model.add(Dense(128,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()
#손실함수 binary_crossentropy 
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#binary_crossentropy -> sparse_categorical_crossentropy 로변경, 왜 와이? 아웃풋이 2개니까 
history = model.fit(x_train, y_train, epochs=100, batch_size=10,validation_data=(x_val, y_val))
# model.save('is_sleeping.h5')


# # #test 데이터 돌려서 plotting 
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
test_prediction = np.argmax(model.predict(x_test), axis=-1)

# #plt로 accuracy 및 validation 
plt.plot(history.history['accuracy']) #훈련 정확도 
plt.plot(history.history['val_accuracy'])
plt.title('model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

 

'''
# 훈련 과정 시각화 (손실)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''





