'''Detection of iris and  drawing on screen'''
import cv2
import numpy as np
from random import randint
import math
import time
import threading, queue
import serial


def readSerial(flag, tensao, q):
    count = 0
    now = int(time.time())
    with serial.Serial('/dev/ttyUSB0', 9600, timeout=1) as ser:
        while(1):
            prox = ser.read(1).hex()
            if(prox == 'ff'):
                prox = ser.read(1).hex()
                if(prox == '00'):
                    prox = ser.read(1).hex()
                    if(prox == 'ff'):
                        prox = ser.read(1).hex()
                        if(prox == '01'):
                            
                            # 03ff     -> 1023
                            tensao = int(ser.read(2).hex(), 16)*(5/1023) #mapeando para 1023 sendo 5V
                            count += 1
                            #tempo = int(ser.read(4).hex(), 16)/1000 #tempo
                            #print(tensao)
                            # if tensao > 2:
                            #     print(tensao)
                            #     flag = True
                            #     q.put(flag)
                            # else:
                            #     pass
                            # print('Tensao na thread: {}' .format(tensao))
                            if(tensao >= 1.97):
                                flag = True
                                q.put(flag)
                                q.put(tensao)
                            if count == 1000:
                                print(int(time.time()) - now)
                                count = 0
                                now = int(time.time())
                            #if tensao > 2.8:
                            #    acionadas += 1
                            #    print(acionadas)
                            #    time.sleep(0.5)
                            #print(tensao)
                            #comenta aki
                            #ys.append(tensao)
                            #xs.append(tempo)
                            #if len(ys) > 50:
                            #    break
                            #if(cont > 30):
                            #    del ys[0]
                            #else:
                            #    cont = cont + 1
                            #ser.flush()    
      


font = cv2.FONT_HERSHEY_SIMPLEX

def translate_coord(pos_x, pos_y):
    '''Find the correct (px,py) to the new origin point(42, 34)'''
    p_x = 0
    p_y = 0
    center_x = 42
    center_y = 34

    p_x = pos_x - center_x
    p_y = pos_y - center_y
    # print('\nCoordenadas olho: X={} Y={}\n' .format(p_x, p_y))

    angle_rad = np.arctan2(p_y, p_x)
    angle_degree = np.rad2deg(angle_rad)
    if angle_degree < 0:
        angle_degree += 360
        # print('\nGraus ajustado={}' .format(angle_degree))
    # else:
        # print('\nGraus={}'.format(angle_degree))
    
    return p_x, p_y, angle_degree

def find_new_length(pos_x, pos_y, angle_degree):
    '''Find the new ratio aspect for the triangle in bigger screen'''
    p_x = pos_x
    p_y = pos_y
    angle = angle_degree
    center_x = 320
    center_y = 176
    max_x = center_x/2
    max_y = center_y/2
    old_x = 42
    old_y = 34
    ratio_x = (p_x/42)
    ratio_y = (p_y/34)
    new_x = int(center_x*ratio_x)
    new_y = int(center_y*ratio_y)
    # print('Old X={} Y={}' .format(p_x, p_y))

    # print('New X={} Y={}' .format(new_x, new_y))

    return new_x, new_y

def main(flag, tensao, q):
    '''Main program function'''
    cap = cv2.VideoCapture('video4.mp4')

    width = int(cap.get(3))
    height = int(cap.get(4))

    init_x, center_x = int(width/2), int(width/2)
    init_y, center_y = int(height/2), int(height/2)
    # print(width, height)
    img = np.zeros((height, width, 3), np.uint8)
    while True:

        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([20, 0, 70])
        upper_green = np.array([100, 255, 255])

        # lower_green = np.array([0, 10, 50])
        # upper_green = np.array([150, 150, 150])

        mask = cv2.inRange(frame, lower_green, upper_green)
        res = cv2.bitwise_and(hsv, frame, mask=mask)

        rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = gray[52:120, 334:418]

        # gray = gray[143:180, 350:410]
        cv2.imshow('gray', gray)

        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 30, 255, 0)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # thresh = cv2.erode(thresh,kernel,iterations = 2)
        hsv_new = hsv[52:100, 334:418]
        # hsv_new = hsv[143:180, 350:410]
        # Shape original
        frame_height, frame_width, _ = frame.shape
        # print('FH: {} - FW: {}' .format(frame_height, frame_width))
        # print('\n')

        # Shape resized
        frame_height2, frame_width2 = thresh.shape
        # print('FH-thresh: {} - FW-thresh: {}' .format(frame_height2, frame_width2))
        # print('\n')
        # calculate moments of binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=False)
        for cnt in contours:
            cnt_moment = cv2.moments(thresh)
            area = cv2.contourArea(cnt)
            # calculate x,y coordinate of center
            x, y, w, h = cv2.boundingRect(cnt)
            x_r = int(x+w/2)
            y_r = int(y+h/2)
            cv2.rectangle(frame, (330+x, 52+y), (330+x+w, 52+y+h), (255,0,0), 1)
            
            # cv2.rectangle(frame, (350+x, 143+y), (350+x+w, 143+y+h), (255,0,0), 1)
            c_x = int(cnt_moment["m10"] / cnt_moment["m00"])
            c_y = int(cnt_moment["m01"] / cnt_moment["m00"])
            resized_x = 42
            resized_y = 34
            # print('Coordenadas ret menor: X={} Y={}' .format(c_x, c_y))
            cv2.drawContours(hsv_new, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(frame, ((335+c_x), (52+c_y)), 2, (0, 255, 0), -1)
            # cv2.circle(img, ((330+c_x), (52+c_y)), (randint(0,10)), ((randint(0,255), randint(0, 255), randint(0, 255))), -1)
            
            
            
            dist_eucl = math.sqrt((42-c_x)*(42-c_x) + (34-c_y)*(34-c_y))
            # print('Dist euclidian {}' .format(dist_eucl))
            angle_rad = np.arctan((c_y-34)/(c_x-42))
            angle_degree = np.degrees(angle_rad)
            # print('Angle {}' .format(angle_degree))
            # Relação novo eixo: (POSICAO - X_CENTER, Y_CENTER - POSICAO)
            point_x, point_y, angle_degree = translate_coord(c_x, c_y)
            draw_x, draw_y = find_new_length(point_x, point_y, angle_degree)

            # if flag:
            #     # print(tensao)
            #     tensao_draw = tensao*abs((tensao-1.5))*2
            #     cv2.circle(img, ((init_x+c_x), (init_y+c_y)), q_draw, ((randint(0,255), randint(0, 255), randint(0, 255))), -1)
            try:
                flag_draw = q.get(False)
                tensao_draw = q.get()
                print(tensao_draw)
                value = int(tensao_draw)
                if(value == 1):
                    thickness = 1
                    color = (0, 255, 255)
                elif(value == 2):
                    thickness = 5
                    color = (255, 255, 0)
                else:
                    thickness = 30
                    color = (255, 0, 255)

                if flag_draw:
                    cv2.circle(img, ((init_x+draw_x), (init_y+draw_y)), thickness, color, -1)
            except queue.Empty:
                flag_draw = False
                
            
        
            # if True:
                # print(tensao)
            # cv2.circle(img, ((init_x+draw_x), (init_y+draw_y)), int(tensao_draw), (randint(0, 255), randint(0, 255), randint(0, 255)), -1)

            # Manao
            # if area <= 500:
            #     cv2.drawContours(hsv_new, [cnt], -1, (0, 255, 0), 2)
            #     cv2.circle(frame, ((350+x_r), (143+y_r)), 1, (0, 255, 0), -1)
            #     cv2.circle(img, ((350+c_x*3), (143+c_y*3)), (randint(0,2)), ((randint(0,255), randint(0, 255), randint(0, 255))), -1)
            
            if(((c_x >= 55 )) and (c_y >= 34)):
                # Está olhando direita BAIXO
                cv2.putText(frame,'RIGHT DOWN!',(10,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
            elif(((c_x < 55) and (c_y >= 34))):
                # Está olhando Esquerda BAIXO
                cv2.putText(frame,'LEFT DOWN!',(10,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
            elif(((c_x >= 55) and (c_y < 34))):
                # Está olhando direita cima
                cv2.putText(frame,'RIGHT UP!',(10,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
            elif(((c_x < 55) and (c_y < 34))):
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
        k = cv2.waitKey(40) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

# if __name__ == "__main__":
#     main()
flag = False
tensao = None
q = queue.Queue(maxsize=2)

thread_arduino = threading.Thread(target=readSerial, args=(flag, tensao, q))
thread_arduino.start()
thread_opencv = threading.Thread(target=main, args=(flag, tensao, q))
thread_opencv.start()