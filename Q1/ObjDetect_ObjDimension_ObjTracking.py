import cv2
import numpy as np
import numpy
from matplotlib import pyplot as plt
import tensorflow
import numpy as np
import numba as nb
import depthai as dai

from tensorflow.keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator()

gen = idg.flow_from_directory("./Images",target_size=(200,200))

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Dropout

inlayer = Input(shape=(200,200,3))
c1 = Conv2D(16, 3, activation="relu")(inlayer)
mp1 = MaxPooling2D(2)(c1)
c2 = Conv2D(32, 3, activation="relu")(mp1)
mp2 = MaxPooling2D(2)(c2)
c3 = Conv2D(64, 3, activation="relu")(mp2)
mp3 = MaxPooling2D(2)(c3)
c4 = Conv2D(128, 3, activation="relu")(mp3)
mp4 = MaxPooling2D(2)(c4)
flat = Flatten()(mp4)
d1 = Dense(600, activation="relu")(flat)
d2 = Dense(300, activation="relu")(d1)
d3 = Dense(150, activation="relu")(d2)
d4 = Dense(100, activation="relu")(d3) 
d5 = Dense(50, activation="relu")(d4)
out_layer = Dense(2, activation="softmax")(d5)

model = Model(inlayer, out_layer)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(gen, epochs=1)

streams = []

streams.append('isp')


@nb.njit(nb.uint16[::1] (nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0


    for i in nb.prange(input.size // 5): 
        b4 = input[i * 5 + 4]
        out[i * 4]     = ((input[i * 5]     << 2) | ( b4       & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) |  (b4 >> 6)       ) << lShift

    return out

print("depthai version:", dai.__version__)
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

if 'isp' in streams:
    xout_isp = pipeline.createXLinkOut()
    xout_isp.setStreamName('isp')
    cam.isp.link(xout_isp.input)

device = dai.Device(pipeline)
device.startPipeline()

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=True)
    q_list.append(q)
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (960, 540))



def getClassName(index):
    if index == 0:
        return "object found"
    else:
        return "object not found"

font = cv2.FONT_HERSHEY_COMPLEX
org = (50, 50)
#calculate dimensions to put in the video
foundImg = cv2.imread("./Images/found/capture_isp_1.png")
res,coords = model.predict(np.array([foundImg]))

(fX, fY, fW, fH) = coords

size = "(" + str(fW) + "," + str(fH) + ")"


fontScale = 6

color = (255, 0, 0)  
#
thickness = 2

def img_alignment(img1, img2):
    img1, img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
    img_size = img1.shape
    warp_mode = cv2.MOTION_TRANSLATION

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3,dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3,dtype=np.float32)
    
    n_iterations = 5000
    termination_eps = 1e-10

    criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, n_iterations, termination_eps)

    cc, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(img2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img2_aligned = cv2.warpAffine(img2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return img2_aligned

capture_flag = False
img_counter = 0
while True:
    for index,q in enumerate(q_list):
        name = q.getName()
        data = q.get()
        if index+1 < len(q_list):
            name1 = q_list[index+1].getName()
        else:
            name1 = q_list[index].getName()
        
        if index+1 < len(q_list):
            data1 = q_list[index+1].get()
        else:
            data1 = q_list[index].get()
        width, height = data.getWidth(), data.getHeight()
        width1,height1 = data1.getWidth(),data1.getHeight()

        payload = data.getData()
        payload1 = data1.getData()
        capture_file_info_str = ('capture_' + name
                                #  + '_' + str(width) + 'x' + str(height)
                                 + '_' + str(data.getSequenceNum())
                                )
        capture_file_info_str = f"capture_{name}_{img_counter}"
        capture_file_info_str1 = f"capture_{name}_{img_counter + 1}"
        if name == 'isp':
            shape = (height * 3 // 2, width)
            yuv420p = payload.reshape(shape).astype(np.uint8)
            yuv420p1 = payload1.reshape(shape).astype(np.uint8)
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
            bgr1 = cv2.cvtColor(yuv420p1, cv2.COLOR_YUV2BGR_IYUV)
            grayscale_img =  cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            grayscale_img2 = cv2.cvtColor(bgr1,cv2.COLOR_BGR2GRAY)
        if capture_flag:  
            filename = capture_file_info_str + '.png'
            print("Saving to file:", filename)
            grayscale_img = np.ascontiguousarray(grayscale_img)  
            img2 = np.ascontiguousarray(img2) 
            cv2.imwrite(filename, grayscale_img)
        bgr = np.ascontiguousarray(bgr)  # just in case
        res = model.predict(np.array([cv2.resize(bgr, (200,200))]))
        bgr = cv2.putText(bgr, getClassName(res.argmax(axis=1)) , org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        if getClassName(res.argmax(axis=1))=="object found":
            bgr = cv2.putText(bgr,size,(100,100), font,fontScale, color, thickness, cv2.LINE_AA)

        diff = cv2.absdiff(bgr, bgr1)
        
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        diff_blur = cv2.GaussianBlur(diff_gray, (5,5,), 0)

        _, binary_img = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, b, l = cv2. boundingRect(contour)
            if cv2.contourArea(contour) > 300:
                cv2.rectangle(bgr, (x, y), (x+b, y+l), (0,255,0), 2)
        cv2.imshow(name, bgr)
   
    capture_flag = False
    key = cv2.waitKey(5)
    if key%256 == 27:
       
        print("Operation over")
        break
    elif key%256 == 32:
        capture_flag = True
        img_counter += 1
