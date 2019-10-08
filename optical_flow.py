import cv2
import numpy as np
import time

cap = cv2.VideoCapture('vide.avi')
# Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Lucas kanade params
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point.append((x, y))
        point_selected[-1]=True
        point_selected.append(False)
        old_points.append(np.array([[x, y]], dtype=np.float32))


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)
point_selected = [False]
point = []
old_points =[]
n=1
a=time.time()
while True:
    #print(point)
    #print(point_selected)
    #print(old_points)
    #time.sleep(0.05)
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    key = cv2.waitKey(1)
    if key==ord('p'):
        while cv2.waitKey(1)!=ord('q'):
            for i in range(len(point_selected)):
                if point_selected[i] is True:
                    print(i)
                    #cv2.circle(frame, point[i], 5, (0, 0, 255), 2)
                    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points[i], None, **lk_params)
                    #print(new_points)
                    x, y = new_points.ravel()
                    old_points[i]=np.array([[x,y]],dtype=np.float32)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            old_gray = gray_frame.copy()
    for i in range(len(point_selected)):
        if point_selected[i] is True:
            #print(i)
            #cv2.circle(frame, point[i], 5, (0, 0, 255), 2)
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points[i], None, **lk_params)
            #print(new_points)
            x, y = new_points.ravel()
            old_points[i]=np.array([[x,y]],dtype=np.float32)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    old_gray = gray_frame.copy()
    cv2.imshow("Frame", frame)

    n=n+1
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()