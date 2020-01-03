import cv2
import os

faceCascade = cv2.CascadeClassifier('Cascades/cascade.xml')
image_path = "./trainpic/"
i = 0
for img_name in os.listdir(image_path):
    img = cv2.imread(image_path + img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
            gray,
            # 该参数需要根据自己训练的模型进行调参
            scaleFactor=1.02,
            # minNeighbors控制着误检测，默认值为3表明至少有3次重叠检测，我们才认为人脸确实存
            minNeighbors=3,
            # 寻找人脸的最小区域。设置这个参数过大，会以丢失小物体为代价减少计算量。
            minSize=(3, 3),
            flags=cv2.IMREAD_GRAYSCALE
        )

    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        i += 1
        dstImg = img[y:y+h, x:x+w]
        cv2.imwrite("./partpic/rectpart" + str(i) + '.jpg', dstImg)

print("find " + str(i) + " pic")
exit(0)


