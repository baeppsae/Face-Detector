import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import time
import pickle
import main # 내가 만든 main.py


def loadObjects(file_name):             # 지정한 파일, file_name에서 읽은 객체를 반환하기
    with open(file_name, "rb") as file:
        return(pickle.load(file))

def get_fileName(s):
    if s.rfind('.') == -1:
        return s
    return s[:s.rfind('.')]

img_file_lst = {}
path = main.path
img_lst = main.img_lst

for i in range(main.cnt_img):
    img = cv2.imread(path+img_lst[i])
    img = image.imread(path+img_lst[i])
    img_file_lst[img_lst[i]] = img
    #cv2.namedWindow(img_lst[i])

    file_name = get_fileName(img_lst[i])+'.bin'
    loaded_pickle = loadObjects(file_name)

    for element_of_pickle in loaded_pickle:
        loaded_img = element_of_pickle[0]
        s_x = element_of_pickle[1]
        s_y = element_of_pickle[2]
        e_x = element_of_pickle[3]
        e_y = element_of_pickle[4]

        cv2.rectangle(img_file_lst[img_lst[i]], (s_x, s_y), (e_x, e_y), (0, 255, 0),
                     3)  # 피클에 저장된 좌표를 토대로 사각형을 만든다. 즉, 사용자가 ROI로 선택했던 영역을 다시 그린다
    #cv2.imshow(img_lst[i], img_file_lst[img_lst[i]])
    plt.figure(img_lst[i])
    plt.axis('off')
    plt.imshow(img_file_lst[img_lst[i]])

plt.show()
