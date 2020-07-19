import cv2
import numpy as np

HAAR_FILE="haarcascade_frontalface_default.xml"
cascade=cv2.CascadeClassifier(HAAR_FILE)

img=cv2.imread("image_wave/image_wave-000000.png")

face=cascade.detectMultiScale(img)

for x,y,w,h in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)

    #おでこ部分
    xx=int(x+(5/9)*w)
    yy=int(y+(1/13)*h)
    zz=int(x+(6/9)*w)
    ww=int(y+(2/13)*h)
    #おでこ部分を四角で囲む
    cv2.rectangle(img,(xx,yy),(zz,ww),(0,255,0),1)
    # 対象範囲を切り出し（おでこ）
    boxFromX = xx #対象範囲開始位置 X座標
    boxFromY = yy #対象範囲開始位置 Y座標
    boxToX = zz #対象範囲終了位置 X座標
    boxToY = ww #対象範囲終了位置 Y座標

    # y:y+h, x:x+w　の順で設定
    imgBox = img[boxFromY: boxToY, boxFromX: boxToX]

    # RGB平均値を出力
    # flattenで一次元化しmeanで平均を取得
    b = imgBox.T[0].flatten().mean()
    g = imgBox.T[1].flatten().mean()
    r = imgBox.T[2].flatten().mean()

    # RGB平均値を取得
    #print("B: %.2f" % (b))
    cv2.putText(img,"%.2f" % (g), (300, 195), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), lineType=cv2.LINE_AA)

    cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
