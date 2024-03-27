import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import main # 1번 코드 이름으로 변경해주면 됨

lst_of_xy = {}
img_file_lst = {}
img_closed = {}
closed_cnt = 0
path = main.path
img_lst = main.img_lst
image = None # 현재 선택중인 이미지
win_name = '' # 현재 선택중인 창 이름
mouse_pressed = False
s_x = 0
s_y = 0
e_x = 0
e_y = 0

def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed, drawing_needed, image, win_name
    if event == cv2.EVENT_RBUTTONDOWN:
        win_name = param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        win_name = param[1]
        image = np.copy(img_file_lst[win_name])
        s_x, s_y = x, y             # 선택 시작 좌표 기록
        #image_to_show = np.copy(image)     # 불 필요. 삭제함.

    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동중이다...
        if mouse_pressed:               # 누른 상태에서...
            img_file_lst[win_name] = np.copy(image)      # 녹색 사각형을 지워야 함.
            cv2.rectangle(img_file_lst[win_name], (s_x, s_y),
                          (x, y), (0, 255, 0), 1)       # 녹색 사각형으로 선택 영역 보이기

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y             # 선택 종료 좌표 기록
        cv2.rectangle(img_file_lst[win_name], (s_x, s_y), (e_x, e_y), (0, 255, 0), 1)
        if win_name not in lst_of_xy.keys():
            lst_of_xy[win_name] = []
        lst_of_xy[win_name].append([param[0][s_y: e_y, s_x: e_x], s_x, s_y, e_x, e_y])
        drawing_needed = True

def saveObjects(file_name, data):       # data를 지정한 이름의 파일 file_name에 저장하기
    with open(file_name, "wb") as file:
        pickle.dump(data, file)

def get_fileName(s):
    if s.rfind('.') == -1:
        return s
    return s[:s.rfind('.')]

for i in range(main.cnt_img):
    img = cv2.imread(path+img_lst[i])
    img_file_lst[img_lst[i]] = img
    img_closed[img_lst[i]] = False
    cv2.namedWindow(img_lst[i])
    cv2.setMouseCallback(img_lst[i], mouse_callback, param=[img_file_lst[img_lst[i]], img_lst[i]])

while True:
    for i in range(main.cnt_img):
        if not img_closed[img_lst[i]]:
            cv2.imshow(img_lst[i], img_file_lst[img_lst[i]])

    k = cv2.waitKey(1)

    if k == ord(' '):
        if win_name == '':
            continue
        file_name = get_fileName(win_name) + '.bin'
        saveObjects(file_name, lst_of_xy[win_name])
        cv2.destroyWindow(win_name)
        img_closed[win_name] = True
        closed_cnt += 1
        win_name = ''
    if closed_cnt == main.cnt_img:
        break
