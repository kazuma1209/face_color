import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


HAAR_FILE="haarcascade_frontalface_default.xml"
cascade=cv2.CascadeClassifier(HAAR_FILE)

cap = cv2.VideoCapture(0)



while True:

    ret, frame = cap.read()
    face=cascade.detectMultiScale(frame)

    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)

        xx=int(x+(5/9)*w)
        yy=int(y+(1/13)*h)
        zz=int(x+(6/9)*w)
        ww=int(y+(2/13)*h)

        cv2.rectangle(frame,(xx,yy),(zz,ww),(0,255,0),1)

        cv2.rectangle(frame,(700,300),(750,320),(255,0,0),1)

        cv2.imshow('frame', frame)
        # 対象範囲を切り出し
        boxFromX = xx #対象範囲開始位置 X座標co1
        boxFromY = yy #対象範囲開始位置 Y座標in1
        boxToX = zz #対象範囲終了位置 X座標co2
        boxToY = ww #対象範囲終了位置 Y座標in2

        # y:y+h, x:x+w　の順で設定
        imgBox = frame[boxFromY: boxToY, boxFromX: boxToX]

        # RGB平均値を出力
        # flattenで一次元化しmeanで平均を取得
        b = imgBox.T[0].flatten().mean()
        g = imgBox.T[1].flatten().mean()
        r = imgBox.T[2].flatten().mean()

        # RGB平均値を取得
        #print("B: %.2f" % (b))
        print("%.2f" % (g))



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
