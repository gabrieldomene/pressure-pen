'''Detection of iris and  drawing on screen'''
import cv2
import numpy as np
from random import randint

font = cv2.FONT_HERSHEY_SIMPLEX

def translate_coord(value, left_min, left_max, right_min, right_max):
    '''Map coordinates to draw on screen'''
    pass

def main():
    '''Main program function'''
    cap = cv2.VideoCapture('video4.mp4')

    width = int(cap.get(3))
    height = int(cap.get(4))

    # print(width, height)
    img = np.zeros((height, width, 3), np.uint8)
    while True:

        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([20, 0, 70])
        upper_green = np.array([100, 255, 255])

        mask = cv2.inRange(frame, lower_green, upper_green)
        res = cv2.bitwise_and(hsv, frame, mask=mask)

        rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = gray[52:120, 334:418]
        cv2.imshow('gray', gray)

        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 30, 255, 0)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # thresh = cv2.erode(thresh,kernel,iterations = 2)
        hsv_new = hsv[52:100, 334:418]
        # Shape original
        frame_height, frame_width, _ = frame.shape
        print('FH: {} - FW: {}' .format(frame_height, frame_width))
        print('\n')

        # Shape resized
        frame_height2, frame_width2 = thresh.shape
        print('FH-thresh: {} - FW-thresh: {}' .format(frame_height2, frame_width2))
        print('\n')
        # calculate moments of binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
            cnt_moment = cv2.moments(thresh)
            # calculate x,y coordinate of center
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (330+x, 52+y), (330+x+w, 52+y+h), (255,0,0), 1)
            c_x = int(cnt_moment["m10"] / cnt_moment["m00"])
            c_y = int(cnt_moment["m01"] / cnt_moment["m00"])
            print(c_y, c_x)
            cv2.drawContours(hsv_new, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(frame, ((335+c_x), (52+c_y)), 5, (0, 255, 0), -1)
            cv2.circle(img, ((330+c_x*3), (52+c_y*3)), (randint(0,2)), ((randint(0,255), randint(0, 255), randint(0, 255))), -1)
            if(((c_x >= 55 )) and (c_y >= 24)):
                # Está olhando direita BAIXO
                cv2.putText(frame,'RIGHT DOWN!',(10,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
            elif(((c_x < 55) and (c_y >= 24))):
                # Está olhando Esquerda BAIXO
                cv2.putText(frame,'LEFT DOWN!',(10,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
            elif(((c_x >= 55) and (c_y < 24))):
                # Está olhando direita cima
                cv2.putText(frame,'RIGHT UP!',(10,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
            elif(((c_x < 55) and (c_y < 24))):
                # Está olhando esquerda cima
                cv2.putText(frame,'LEFT UP!',(10,50), font, 1, (0,255, 0), 2, cv2.LINE_AA)
            break

        cv2.imshow('threshold', thresh)
        # roi = res[52:100, 334:418]
        cv2.imshow('HSV NEW', hsv_new)
        # cv2.imshow('Resolution', res)

        # roi = frame[52:100, 334:418]
        # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # equ = cv2.equalizeHist(gray_roi)
        # equ = cv2.GaussianBlur(equ, (7, 7), 0)


        # _, thresh = cv2.threshold(equ, 100, 255, 1)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for cnt in contours:
        #     cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)


        # cv2.imshow('Equalize', equ)
        # cv2.imshow('Thresh', thresh)
        # cv2.imshow('Footage', frame)
        # cv2.imshow('ROI', roi)

        cv2.imshow('Black', img)
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
