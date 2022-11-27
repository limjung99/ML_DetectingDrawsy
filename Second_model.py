from First_model import *

# (50개의 눈 사진 ) 들의 배열  
# load data

with open('videos_array2', 'rb') as fr:
    videos_arr = pickle.load(fr)
fr.close()
with open("video_labels2","rb") as f:
    ans = pickle.load(f)
f.close()

#깨어있으면 1 자고있으면 0 

# with open("videos_array2","wb") as va:
#     pickle.dump(vid_arr,va)
#     va.close()
# with open("video_labels2","wb") as vl:
#     pickle.dump(vid_label, vl)
#     vl.close()


predicted = [] #예측한 눈동자 상태 배열이 안에 append 된다.
first_model = model #CNN모델 import 


if os.path.isfile("predicted"):
    with open("predicted","rb") as f:
        predicted = pickle.load(f)
    f.close()
else:
    for i in videos_arr:
        predicted.append(np.argmax(first_model.predict(i),axis=-1))
    with open("predicted","wb") as f:
        pickle.dump(predicted,f)
    f.close()



# #train 및 test데이터 분리 
x_train, x_test, y_train, y_test = train_test_split(np.array(predicted), np.array(ans), test_size=0.2)
# validation 데이터 분리 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)


# Define the K-fold Cross Validator
#kfold = KFold(n_splits=5, shuffle=True)

#K-fold Cross Validation model evaluation
#fold_no = 1

model_file = './fitted_second_model.h5'
if os.path.isfile(model_file):
    second_model = load_model('fitted_second_model.h5')
else:
    #dataset이 부족하므로, k fold validation으로 검증 
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