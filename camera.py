import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # FRAME IS ALREADY IN RGB

    # Our operations on the frame come here

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the resulting frame
    #cv2.imshow('RGB',rgb)
    #cv2.imshow('HSV',hsv)

    rgb_threshold = cv2.inRange(rgb, (100, 150, 150), (255, 255, 255))
    hsv_threshold0 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    hsv_threshold1 = cv2.inRange(hsv, (160, 70, 70), (180, 255, 255))

    hsv_thresh = hsv_threshold0 + hsv_threshold1

    contours,hierarchy = cv2.findContours(hsv_thresh, 1, 2)
    for contour in contours:
        cnt = contour
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)

    #cv2.imshow("ThreshHSV", hsv_thresh)
    cv2.imshow("Frame w/ Bounding Box", frame)

    image = frame 

    data = cv2.resize(image, (100, 100)).reshape(-1, 3)
    iamge = data
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data.astype(np.float32), 1, None, criteria, 10, flags)

    cluster_sizes = np.bincount(labels.flatten())

    palette = []
    for cluster_idx in np.argsort(-cluster_sizes):
        palette.append(np.full((image.shape[0], image.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
    palette = np.hstack(palette)

    sf = image.shape[1] / palette.shape[1]
    out = np.vstack([image, cv2.resize(palette, (0, 0), fx=sf, fy=sf)])

    cv2.imshow("dominant_colors", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()