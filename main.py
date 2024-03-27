import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import dlib
import face_recognition
import time
import cvlib as cv

path = '../../data/'  # 각 얼굴 검출기와 샘플 사진과 비디오파일들이 있는 폴더

threshold_IOU = [0.5, 0.6, 0.7, 0.8, 0.9]
cnt_IOU = len(threshold_IOU)

img_lst = ['twice2.jpg', 'bts2.jpg', 'vikings_s.jpg',
           'c_370384-l_1-k_imagepuff.jpeg', 'group-happy-friends-taking-photo_1139-267.jpg']
cnt_img = len(img_lst)


def show_detection(image, faces, type):
    # 입력 영상과 얼굴 4각형 정보를 받아 입력 영상에 4각형 얼굴 표시를 하여 반환한다.
    # 입력 영상은 BGR 구성인 것으로 가정.

    for i in range(len(faces)):
        if type == "cv2_haar":
            x, y, w, h = faces[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # (B. G. R)
        elif type == "dlib_HoG":
            face = faces[i]
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 3)
        elif type == "fr_CNN":
            face = faces[i]
            top, right, bottom, left = face
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
        elif type == "cvlib_dnn":
            for (startX, startY, endX, endY) in faces:
                cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

    return image


def loadObjects(file_name):  # 지정한 파일, file_name에서 읽은 객체를 반환하기
    with open(file_name, "rb") as file:
        return (pickle.load(file))


def get_fileName(s):
    if s.rfind('.') == -1:
        return s
    return s[:s.rfind('.')]


def check_overlap(find, ans):
    check1 = False
    check2 = False
    if find[0] <= ans[0] <= find[1]:
        check1 = True
    if ans[0] <= find[0] <= ans[1]:
        check1 = True
    if find[2] <= ans[2] <= find[3]:
        check2 = True
    if ans[2] <= find[2] <= ans[3]:
        check2 = True
    if check1 and check2:
        return True
    return False


def find_TP(faces, threshold, type):
    global ans_cnt
    tp = 0
    for idx in range(len(faces)):
        if type == "cv2_haar":
            x, y, w, h = faces[idx]
            find_x1 = x
            find_x2 = x + w
            find_y1 = y
            find_y2 = y + h
        elif type == "dlib_HoG":
            face = faces[idx]
            find_x1 = face.left()
            find_x2 = face.right()
            find_y1 = face.top()
            find_y2 = face.bottom()
        elif type == "fr_CNN":
            face = faces[idx]
            find_y1, find_x2, find_y2, find_x1 = face
        elif type == "cvlib_dnn":
            find_x1, find_y1, find_x2, find_y2 = faces[idx]
        overlap_lst = []
        # for coord, check in pickle_lst:
        for i in range(len(pickle_lst)):
            if not pickle_lst[i][1]:
                ans_x1, ans_y1, ans_x2, ans_y2 = pickle_lst[i][0]
                if check_overlap([find_x1, find_x2, find_y1, find_y2], [ans_x1, ans_x2, ans_y1, ans_y2]):
                    area_all = 0
                    area_all += (ans_x2 - ans_x1) * (ans_y2 - ans_y1)
                    area_all += (find_x2 - find_x1) * (find_y2 - find_y1)
                    overlap_area = (min(find_x2, ans_x2) - max(find_x1, ans_x1)) * (
                                min(find_y2, ans_y2) - max(find_y1, ans_y1))
                    area_all -= overlap_area

                    IOU = overlap_area / area_all
                    if IOU >= threshold:
                        overlap_lst.append((IOU, i))
        best_IOU = 0
        best_idx = -1
        for i in range(len(overlap_lst)):
            if overlap_lst[i][0] > best_IOU:
                best_idx = overlap_lst[i][1]
        if best_idx != -1:
            pickle_lst[best_idx][1] = True
            ans_cnt -= 1
            tp += 1
    return tp


if __name__ == "__main__":
    detection_lst = ['cv2_haar', 'dlib_HoG', 'fr_CNN', 'cvlib_dnn']
    mAP_lst = {}
    pickle_lst = []
    for i in range(cnt_img):
        img = cv2.imread(path + img_lst[i])
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # dlib_HoG에 사용

        # haar 코드
        retval, faces_haar_alt2 = cv2.face.getFacesHAAR(img,
                                                        path + "cascades/haarcascades/haarcascade_frontalface_alt2.xml")
        faces_haar_alt2 = np.squeeze(faces_haar_alt2)
        img_faces_alt2 = show_detection(img.copy(), faces_haar_alt2, detection_lst[0])

        # HoG 코드
        detector = dlib.get_frontal_face_detector()
        faces_HoG = detector(gray_img, 1)
        img_faces_HoG = show_detection(img.copy(), faces_HoG, detection_lst[1])

        # CNN 코드
        faces_CNN = face_recognition.face_locations(img.copy(), 1, "cnn")
        img_faces_CNN = show_detection(img.copy(), faces_CNN, detection_lst[2])

        # cvlib_dnn 코드
        faces_dnn, confidences = cv.detect_face(img)
        img_faces_dnn = show_detection(img.copy(), faces_dnn, detection_lst[3])

        file_name = get_fileName(img_lst[i]) + '.bin'
        loaded_pickle = loadObjects(file_name)
        pickle_lst = []
        for element_of_pickle in loaded_pickle:
            loaded_img, s_x, s_y, e_x, e_y = element_of_pickle
            pickle_lst.append([[s_x, s_y, e_x, e_y], False])  # 사용된 정답을 처리하기 위해 필요
            ''' 이미지 출력 관련
            cv2.rectangle(img_faces_alt2, (s_x, s_y), (e_x, e_y), (0, 255, 0),
                          3)  # 피클에 저장된 좌표를 토대로 사각형을 만든다. 즉, 사용자가 ROI로 선택했던 영역을 다시 그린다
            cv2.rectangle(img_faces_HoG, (s_x, s_y), (e_x, e_y), (0, 255, 0),
                          3)  # 피클에 저장된 좌표를 토대로 사각형을 만든다. 즉, 사용자가 ROI로 선택했던 영역을 다시 그린다
            cv2.rectangle(img_faces_CNN, (s_x, s_y), (e_x, e_y), (0, 255, 0),
                          3)  # 피클에 저장된 좌표를 토대로 사각형을 만든다. 즉, 사용자가 ROI로 선택했던 영역을 다시 그린다
            cv2.rectangle(img_faces_dnn, (s_x, s_y), (e_x, e_y), (0, 255, 0),
                          3)  # 피클에 저장된 좌표를 토대로 사각형을 만든다. 즉, 사용자가 ROI로 선택했던 영역을 다시 그린다
            '''

        faces = [faces_haar_alt2, faces_HoG, faces_CNN, faces_dnn]

        for idx in range(len(faces)):
            sum_Recall_Accuracy = 0
            if detection_lst[idx] not in mAP_lst.keys():
                mAP_lst[detection_lst[idx]] = 0
            for j in range(cnt_IOU):
                ans_cnt = len(pickle_lst)  # FN을 판정할 때 사용 예정
                tp = find_TP(faces[idx], threshold_IOU[j], detection_lst[idx])
                sum_Recall_Accuracy += tp / (tp + ans_cnt)
                for k in range(len(pickle_lst)):
                    pickle_lst[k][1] = False
                # print(img_lst[i], threshold_IOU[j], tp)
            mAP_lst[detection_lst[idx]] += sum_Recall_Accuracy

        ''' 이미지 출력부
        img_RGB = [img_faces_alt2[:, :, ::-1], img_faces_HoG[:, :, ::-1], img_faces_CNN[:, :, ::-1],
                   img_faces_dnn[:, :, ::-1]]

        plt.figure(img_lst[i])

        for j in range(4):
            plt.subplot(2, 2, j + 1)
            plt.imshow(img_RGB[j])
            plt.title(detection_lst[j])
            plt.axis('off')
    
    plt.show()
    '''
    print("추가 정보:\nOpenCV_Haar : Color영상 사용, alt2사용\ndlib_HoG : up_scale=1")
    print("faceRecognition_CNN : scale=1\ncvlib_DNN : 특이사항 없음")

    for i in range(len(detection_lst)):
        print(detection_lst[i], "의 mAP :", mAP_lst[detection_lst[i]] / (cnt_IOU * cnt_img))

    plt.figure("mAP Graph")
    plt.title("mAP Graph")
    x = np.arange(len(detection_lst))
    value = []
    for i in range(len(mAP_lst)):
        value.append(mAP_lst[detection_lst[i]] / (cnt_IOU * cnt_img))
    plt.bar(x, value, width=0.8)
    plt.xticks(x, detection_lst)
    plt.ylim(0, 1)
    plt.show()
