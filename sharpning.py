import cv2
import numpy as np
import dlib

predictor = dlib.shape_predictor("./preTrainedModel/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

img = cv2.imread('/home/jwsi/beautyGAN-tf-Implement-master/IDPhoto/testA/shlee.jpg')
img  = cv2.resize(img, (256,256))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
img = cv2.filter2D(img, -1, sharpening_mask1)

dets = detector(gray, 1)
lip_mask = np.zeros([img.shape[0],img.shape[1] ])
    
for face in dets:
    shape = predictor(img, face)
    temp = []
    

    for pt in shape.parts():
        temp.append([pt.x, pt.y])
    

    #face_mask = np.full((256, 256), 255).astype(np.uint8)
    cv2.fillPoly(lip_mask, [np.array(temp[48:60]).reshape((-1, 1, 2))], (255, 255, 255))
    cv2.fillPoly(lip_mask, [np.array(temp[60:68]).reshape((-1, 1, 2))], (0, 0, 0))

    cv2.fillPoly(lip_mask, [np.array(temp[36:42]).reshape((-1, 1, 2))], (255, 255, 255))
    cv2.fillPoly(lip_mask, [np.array(temp[42:47]).reshape((-1, 1, 2))], (255, 255, 255))
    cv2.fillPoly(lip_mask, [np.array(temp[27:36]).reshape((-1, 1, 2))], (255, 255, 255))
     
while True:
    cv2.imshow('imgss', img)
    cv2.imshow('imgss2', gray)
    cv2.imshow('imgss3', lip_mask)


    cv2.waitKey(1)
