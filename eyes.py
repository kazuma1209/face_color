import cv2 as cv
import numpy as np
import dlib

print('cv version is ', cv.__version__)
print('numpy version is ', np.__version__)
print('dlib version is', dlib.__version__)

#ビデオのセッティングと定義
cap = cv.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
showSep = True
showMidResult = True
lastLeftOpen = 1
curLeftOpen = 1
lastRightOpen = 1
curRightOpen = 1
leftBlinkCount = 0
rightBlinkCount = 0

def eye_aspect_ratio(eye):
    # compute the distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    dv1 = pow(pow(eye[1].x - eye[5].x, 2) + pow(eye[1].y - eye[5].y, 2), 0.5)
    dv2 = pow(pow(eye[2].x - eye[4].x, 2) + pow(eye[2].y - eye[4].y, 2), 0.5)
    # compute the distance between the horizontal
    # eye landmark (x, y)-coordinates
    dl3 = pow(pow(eye[0].x - eye[3].x, 2) + pow(eye[0].y - eye[3].y, 2), 0.5)
    # compute and return the eye aspect ratio
    return (dv1 + dv2) / dl3

while cap.isOpened():
    # 1. Take each frame
    # ret, frame = cap.read()
    cap.grab()
    ret, frame = cap.retrieve()
    if np.shape(frame) == ():
        continue

    # flip left and right if your camera need
    frame = cv.flip(frame, 1)

    if showSep:
        cv.namedWindow("camera image", 0)
        cv.imshow("camera image", frame)
    else:
        cv.destroyWindow("camera image")

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # if click A or a
    if k == 65 or k == 97:
        showSep = not showSep

    if k == 65+2 or k == 97 + 2:
        showMidResult = not showMidResult

    # 2. detect face
    dets = detector(frame[:, :, ::-1])
    if len(dets) < 1:
        cv.putText(frame, "Please show more than half face {:d}".format(len(dets)), (0, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv.namedWindow("blink check", 0)
        cv.imshow("blink check", frame)
        continue

    parts = predictor(frame, dets[0]).parts()
    img = frame
    if showSep:
        img = frame * 0

    for i in parts:
        cv.circle(img, (i.x, i.y), 3, (255, 0, 0), -1)

    # 3. find eye here
    # eye 1
    leftEye = [parts[36], parts[37], parts[38], parts[39], parts[40], parts[41]]
    for i in leftEye:
        cv.circle(img, (i.x, i.y), 3, (0, 255, 0), -1)
    # eye 2
    rightEye = parts[42:48]
    for i in rightEye:
        cv.circle(img, (i.x, i.y), 3, (0, 0, 255), -1)

    if showSep:
        cv.namedWindow("detected", 1)
        cv.imshow("detected", img)
    else:
        cv.destroyWindow("detected")

    # 4. check blink
    checkLeft = eye_aspect_ratio(leftEye)
    lineColor = (0, 255, 0)

    if checkLeft < 0.3:
        curLeftOpen = 0
        lineColor = (0, 0, 255)
    else:
        curLeftOpen = 1
        lineColor = (0, 255, 0)

    if lastLeftOpen == 1 and curLeftOpen == 0:
        leftBlinkCount += 1
    lastLeftOpen = curLeftOpen

    for i in leftEye:
        cv.circle(img, (i.x, i.y), 3, lineColor, -1)

    cv.putText(img, "Left Blinks: {}".format(leftBlinkCount), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, lineColor, 2)
    cv.putText(img, "Eye ratio: {:.2f}".format(checkLeft), (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, lineColor, 2)

    print("left", checkLeft, leftBlinkCount, '\n')

    # eye 2
    checkRight = eye_aspect_ratio(rightEye)

    lineColor = (0, 255, 0)
    if checkRight < 0.3:
        curRightOpen = 0
        lineColor = (0, 0, 255)
    else:
        curRightOpen = 1
        lineColor = (0, 255, 0)

    if curRightOpen == 0 and lastRightOpen == 1:
        rightBlinkCount += 1
    lastRightOpen = curRightOpen

    for i in rightEye:
        cv.circle(img, (i.x, i.y), 3, lineColor, -1)

    print("right", checkRight, rightBlinkCount, '\n')

    cv.putText(img, "Right Blinks: {}".format(rightBlinkCount), (300, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, lineColor, 2)
    cv.putText(img, "Eye ratio: {:.2f}".format(checkRight), (300, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, lineColor, 2)

    # 5. show out result
    cv.namedWindow("blink check", 1)
    cv.imshow("blink check", img)

cap.release()
cv.destroyAllWindows()
