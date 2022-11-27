import cv2, dlib
import numpy as np
from imutils import face_utils
import time
import First_model,Second_model

IMG_SIZE = (100, 100)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# 눈 이미지 및 눈 표현할 사각형 표시
def crop_eye(gray, eye_points): 
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)
    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
    return eye_img, eye_rect

#계속 크롭한 이미지를 prediction함수에 넣어 현재 졸고 있는지 예측 후 return
def get_drawsy(crop_images):
    images_prediction = First_model.blink_prediction(crop_images)
    now_drawsy = Second_model.drawsy_prediction(images_prediction)
    return now_drawsy[0] 

# main
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    count = 0 # 캡처한 이미지 수
    images_array = [] # 실제 이미지 array가 들어갈 list
    drawsy_array = [] # 현재 졸음 상태를 저장할 array
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Drawsy Detected!!!"
    is_drawsy = 0 # 졸음 상태인지 아닌지 확인
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    # 5초에 50장 뽑아냄
    while True:
        time.sleep(0.1)
        ret, img_ori = cap.read()
        if not ret:
            break

        img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

        img = img_ori.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector(gray)
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2
        for face in faces:
            shapes = predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)
            x = face.left()
            y = face.top()
            w = face.right() - face.left()
            h = face.bottom() - face.top()

            eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
            eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])
            ex,ey,ew,eh = eye_rect_l
            eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)
            if(is_drawsy == 1):
                cv2.putText(img, text, (textX, textY), font, 1, (0, 0, 255), 2)
            # 눈 인식하면 이를 이미지에 저장하고 직사각형으로 표시
            cv2.rectangle(img, (ex,ey), (ew,eh), color=(255,255,255), thickness=1)
            images_array.append(eye_img_l)
            count+=1 
            # print(img[ey:eh,ex:ew,:].shape)
        cv2.imshow('result2', img)
        if cv2.waitKey(1) == ord('q'):
            break
        # 7초가 되면 0~5초,1~6초,2~7초에 찍은 50장씩의 데이터를 prediction하여 졸음 인식
        if count %70 == 0:
            for i in range(0,3):
                start = i * 10
                end = start + 50
                crop_images = np.array(images_array[start:end])
                now_drawsy = get_drawsy(crop_images)
                drawsy_array.append(now_drawsy)
            if sum(drawsy_array) > 1:
                is_drawsy = 0
            else:
                is_drawsy = 1
            images_array = []
            count = 0
            drawsy_array = []
# images_array => 5초에 50장 저장한 배열

cv2.destroyAllWindows()