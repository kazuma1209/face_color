import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import seaborn as sns
import os
import shutil
from scipy import signal
import glob
from scipy import fftpack

#フレーム分割
def frame_split(video_file='./image_wave.mov', image_dir='./image_wave/',
                   image_file='image_wave-%s.png'):
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    i = 0

    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if flag == False:
            break
        cv2.imwrite(image_dir + image_file % str(i).zfill(6),
                    frame)
        print('Save', image_dir + image_file % str(i).zfill(6))
        i += 1

    cap.release()

frame_split("img3.mov")

#画像ファイルを格納，時系列にソート
files = glob.glob("image_wave/*.png")
files.sort()

#画像を読み込み，green成分を抽出
images = [cv2.imread(files[i]) for i in range(len(files))]
images_green = [pd.DataFrame(images[j][:,:,1]) for j in range(len(images))]

#parameter
in1 = 1100
in2 = 1200
co1 = 200
co2 = 250

#メディアンフィルタによる平滑化（iloc[行番号x,列番号y]）
images_green_median = [cv2.medianBlur(images_green[i].iloc[ in1 : in2 , co1 :co2 ].values,ksize=5)  for i in range(len(images_green))]

#平均値を算出
img_mean = [images_green_median[i].mean()  for i in range(len(images_green_median))]

#サンプリング周波数
fs = 30

end_time = round(len(img_mean)/fs)
time = np.arange(0 , end_time , end_time/len(img_mean))
plt.plot(time, img_mean)
plt.xlabel("time(s)",size=15)
plt.show()
