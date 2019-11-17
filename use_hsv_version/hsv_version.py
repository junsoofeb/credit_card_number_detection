import cv2 as cv
import time
import numpy as np

protoFile = "./file/pose_deploy.prototxt"
weightsFile = "./file/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

ROI = None
target = None
target_hsv = None 

def show(frame):
    cv.imshow("", frame)
    cv.waitKey()
    #cv.destroyAllWindows()

def make_hsv(action_frame):
    blur = cv.GaussianBlur(action_frame, (3,3), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_RGB2HSV)

    lower_color = np.array([108, 23, 82])
    upper_color = np.array([179, 255, 255])

    mask = cv.inRange(hsv, lower_color, upper_color)
    blur = cv.medianBlur(mask, 5)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    hsv = cv.dilate(blur, kernel)

    return hsv

def hand_point():
    global target
    #frame = cv.imread("./ROI.jpg")
    frame = target.copy()
    frameCopy = np.copy(frame)
    frameCopy = cv.resize(frameCopy, (500, 500))
    
    hsv = make_hsv(frameCopy)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight
    threshold = 0.1

    t = time.time()
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    #print("time taken by network : {:.3f}".format(time.time() - t))

    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        probMap = cv.resize(probMap, (500, 500))

        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)

        if prob > threshold :
            cv.circle(frameCopy, (int(point[0]), int(point[1])), 3, (0, 255, 255), thickness=-1, lineType=cv.FILLED)
            cv.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv.LINE_AA)
            #print(i, "번째 포인트의 x, y 좌표 ({}, {})".format(point[0], point[1]))

            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)
    
    cv.imwrite('Output-Keypoints.jpg', frameCopy)

    show(frameCopy)    

    #print("Total time taken : {:.3f}".format(time.time() - t))
    if points[4] == None or points[8] == None :
        print("points is NONE")
        return
    
    if abs(points[4][0] - points[8][0]) > 50:
        print("points error")
        return
        
    return points


def capture():
    global target
    global target_hsv
    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv.resize(frame, (500,500))
        hsv = make_hsv(frame)
        cv.imshow("original image", frame)
        cv.imshow("hsv image", hsv)
        if cv.waitKey(1) & 0xFF == ord('s'):
            target = frame
            target_hsv = hsv
            cv.destroyAllWindows()
            cap.release()
            break
            

while 1:
    capture()
    h_point = hand_point()
    if h_point == None:
        continue
    # h 는 y 값의 차이
    h = h_point[4][1] - h_point[8][1]
    w = h 
    try:
        #ROI = ROI[min_y: h , min_x: w]
        ROI = target[h_point[8][1] : h_point[4][1], h_point[8][0] - int(h * 1.25) : h_point[8][0] + 20]
        show(ROI)
        cv.imwrite("./hsv_card_roi.jpg",ROI)
    except:
        print("ROI error")
        continue
