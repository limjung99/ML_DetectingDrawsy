import First_model
import pickle
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense

# load data
def load_data():
    # 깨어있으면 1 자고있으면 0
    if os.path.isfile("videos_array2"):
        with open('videos_array2', 'rb') as fr:
            videos_arr = pickle.load(fr)
        fr.close()
    else:
        with open("videos_array2","wb") as va:
            pickle.dump(videos_arr,va)
        va.close()
    if os.path.isfile("video_labels2"):
        with open("video_labels2","rb") as f:
            videos_label = pickle.load(f)
        f.close()
    else:
        with open("video_labels2","wb") as vl:
            pickle.dump(videos_label, vl)
        vl.close()
    
    return videos_arr,videos_label
    

def make_predicted():
    predicted = [] #예측한 눈동자 상태 배열이 안에 append 된다.
    if os.path.isfile("predicted"):
        with open("predicted","rb") as f:
            predicted = pickle.load(f)
        f.close()
    else:
        x_train, x_test, y_train, y_test = First_model.load_data()
        epochs = 100
        batch_sizes = 10
        first_model,history = First_model.make_model(x_train, x_test, y_train, y_test,epochs,batch_sizes)
        videos_arr, none = load_data()
        for i in videos_arr:
            predicted.append(np.argmax(first_model.predict(i),axis=-1))
        with open("predicted","wb") as f:
            pickle.dump(predicted,f)
        f.close()

    return predicted


def make_model(predicted,ans):
    # #train 및 test데이터 분리 
    x_train, x_test, y_train, y_test = train_test_split(np.array(predicted), np.array(ans), test_size=0.2)
    # validation 데이터 분리 
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)


    # Define the K-fold Cross Validator
    #kfold = KFold(n_splits=5, shuffle=True)

    #K-fold Cross Validation model evaluation
    #fold_no = 1

    model_file = 'fitted_second_model.h5'
    if os.path.isfile(model_file):
        print("ASDF")
        second_model = load_model('fitted_second_model.h5')
    else:
        #dataset이 부족하므로, k fold validation으로 검증 
        print("여기")
        acc_per_fold=[]
        loss_per_fold=[]
        second_model = Sequential()
        second_model.add(Dense(50, activation='relu', input_shape=(50,))) 
        second_model.add(Dense(50, activation='relu')) 
        second_model.add(Dense(25,activation="relu"))
        second_model.add(Dense(10, activation='relu'))
        second_model.add(Dense(2, activation='softmax'))
        #손실함수 binary_crossentropy 
        second_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #binary_crossentropy -> sparse_categorical_crossentropy 로변경, 왜 와이? 아웃풋이 2개니까 
        history = second_model.fit(x_train, y_train, epochs=100, batch_size=5,validation_data=(x_val,y_val))
        second_model.save('fitted_second_model.h5')
    
    return second_model


def drawsy_prediction(images_prediction):
    videos_arr, videos_label = load_data()
    predicted = make_predicted()
    second_model = make_model(predicted,videos_label)
    now_drawsy = np.argmax(second_model.predict(np.array(images_prediction)),axis=-1)
    print("-------------now_drawsy 반환-------------")
    return now_drawsy
#for train, test in kfold.split(x_train, y_train):
# model.save('is_sleeping.h5')
# model2 = model
# Generate generalization metrics
# scores = model.evaluate(x_train[test], y_train[test], verbose=0)
# print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
# acc_per_fold.append(scores[1] * 100)
# loss_per_fold.append(scores[0])

# Increase fold number
# fold_no = fold_no + 1