import cv2
import time
# import acapture
cap = cv2.VideoCapture("/Users/torres/OneDrive/UNB/2020-08 Visão Computacional/Trabalho 1/camera1.webm")

tracker = cv2.TrackerCSRT_create()
tracking = False
lost = False
lost_thres = 1
lost_time = None

def drawBox(img, bbox):
    x, y, w, h = tuple(map(int, bbox))
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 3, 1)

def begin_tracking(img, tracker):
    global tracking
    bbox = cv2.selectROI("Tracking", img, False)
    print(bbox)
    if bbox:
        # tracker = cv2.TrackerCSRT_create()
        tracker.init(img, bbox)
        tracking = True
        lost = False
        lost_time = None
        return True

while cap.isOpened():
    timer = cv2.getTickCount()
    success, img = cap.read()

    if not success:
        break

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.putText(img, str(fps), (75, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    success, bbox = tracker.update(img)

    if success:
        drawBox(img, bbox)
    elif tracking:
        lost = True
        tracking = False
    elif lost:
        if not lost_time:
            lost_time = time.time()
        else:
            if time.time() - lost_time > lost_thres:
                begin_tracking(img, tracker)



    cv2.imshow("Tracking", img)

    wait_key = cv2.waitKey(1)
    if wait_key > 0:
        print('Wait key was ', wait_key)
        print('After AND: ', wait_key & 0xFF)
        wait_key &= 0xFF

        if wait_key == ord('q'):
            break
        if wait_key == ord('t'):
            begin_tracking(img, tracker)

cap.release()
cv2.destroyAllWindows()
