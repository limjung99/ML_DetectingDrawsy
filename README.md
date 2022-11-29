<img width="1092" alt="image" src="https://user-images.githubusercontent.com/81519350/204428646-777af6d5-372e-4221-a507-e889c82635a1.png">

## 설치방법
```bash
$ git clone https://github.com/limjung99/-_-.git
$ cd -_-
$ pip install -r requirements.txt
$ python main.py
```

## 라이브러리 버전

- matplotlib == 3.6.2
- scikit-learn == 1.1.3
- numpy == 1.23.2
- keras == 2.10.0


## 필요한 Data Download
<a href="https://drive.google.com/drive/folders/1F_orUU4ryFcuhEfT6qy7djC2Be7s8B4H? usp=share_link">파일다운로드</a><br>
- fItted_first_model.h5 : 학습한 1차모델
- fittted_second_model.h5 : 학습한 2차모델
- close_eye.zip: 1차모델 학습을 위한 감은 눈 Data
- open_eye.zip: 1차모델 학습을 위한 뜬 눈 Ddata
- videos_array2 : 2차모델 학습을 위한 영상 Data
- video_labels2 : 2차모델 학습을 위한 labels
- predicted : videos_array2를 미리 predict한 Data shape_predictor_68_face_landmarks.dat : 얼굴 인식을 위한 파일
