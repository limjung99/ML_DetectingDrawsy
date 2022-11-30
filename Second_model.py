import First_model
import pickle
import numpy as np
import os 
from sklearn.model_selection import train_test_split,StratifiedKFold
from keras.models import Sequential, load_model
from keras.layers import Dense

#load data
#이미 저장해둔 video image 50장들의 배열과, 정답 label이 존재하면 directory로부터 불러온다.
#없을경우 다음번 사용을 위해 directory에 저장
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
    
#2차 모델 학습을 위해, Dataset이미지들의 눈 상태를 판정하여 그 배열을 return하는 함수
def make_predicted():
    predicted = [] #예측한 눈의 상태 배열이 안에 append 된다.
    #이미 예측해두었던 dictionary가 directory에 존재한다면 불러온다. 
    if os.path.isfile("predicted"):
        with open("predicted","rb") as f:
            predicted = pickle.load(f)
        f.close()
    else:
        #모델이 존재하지않을경우 모델을 만들고, 판정
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

#2차 학습모델을 만든다
def make_model(predicted,ans):
    # #train 및 test데이터 분리
    x_train, x_test, y_train, y_test = train_test_split(np.array(predicted), np.array(ans), test_size=0.2)
    model_file = 'fitted_second_model.h5'
    if os.path.isfile(model_file):
        second_model = load_model('fitted_second_model.h5')
    else:
        #dataset이 부족하므로, k-fold validation으로 검증 
        acc_per_fold=[]
        loss_per_fold=[]
        k=5
        kfold = StratifiedKFold(n_splits=k,shuffle=True)
        fold_no=1
        for train_idx, test_idx in kfold.split(x_train, y_train):
            second_model = Sequential()
            second_model.add(Dense(50, activation='relu', input_shape=(50,))) 
            second_model.add(Dense(50, activation='relu')) 
            second_model.add(Dense(25,activation="relu"))
            second_model.add(Dense(10, activation='relu'))
            second_model.add(Dense(2, activation='softmax'))
            
            second_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            #train
            history = second_model.fit(x_train[train_idx], y_train[train_idx], epochs=100, batch_size=5)

            scores = second_model.evaluate(x_train[test_idx], y_train[test_idx], verbose=0)
            print(f'Score for fold {fold_no}: {second_model.metrics_names[0]} of {scores[0]}; {second_model.metrics_names[1]} of {scores[1]*100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            fold_no = fold_no + 1
        #평균 손실과 정확도를 console에 log로 남긴다.
        print("K=",k," average accuracy=",sum(acc_per_fold)/len(acc_per_fold))
        print("K=",k," average loss=",sum(loss_per_fold)/len(loss_per_fold))
        second_model.save('fitted_second_model.h5')
    
    return second_model

#최종적으로 졸음인지 깨어있는지 판단한다
def drawsy_prediction(images_prediction):
    videos_arr, videos_label = load_data()
    predicted = make_predicted()
    second_model = make_model(predicted,videos_label)
    now_drawsy = np.argmax(second_model.predict(np.array(images_prediction)),axis=-1)
    print("------------2차모델 predict 결과------------")
    print(now_drawsy)
    return now_drawsy





