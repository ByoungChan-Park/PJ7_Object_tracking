import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Object Detection 활성화
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")

# 차량별 ID 부과
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True : 
    ret, frame = cap.read()
    count += 1
    if not ret : 
        break

    # 차량 프레임 중앙값
    center_points_cur_frame = []

    # 사각 프레임을 통한 차량 탐지
    (class_ids, score, boxes) = od.detect(frame)
    for box in boxes : 
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        # print("FRAME Nº", count, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 이전 프레임과 현재 프레임 비교(두번째 프레임부터 작동)
    if count <= 2 :
        for pt in center_points_cur_frame :
            for pt2 in center_points_prev_frame :
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20 :
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
               
                # 동일 객체가 동일 ID를 갖도록 20픽셀 미만일 경우 ID지속
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # 화면에서 벗어나거나 ID 제거
            if not object_exists:
                tracking_objects.pop(object_id)

        # 객체탐지 사정거리 내 진입 시 새로운 ID 부과
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking objects")
    print(tracking_objects)

    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)
    
    cv2.imshow("Frame", frame)

    # 동일 객체 판단 시 프레임 비교를 위한 포지션 복제
    center_points_prev_frame = center_points_cur_frame.copy()

    # key = cv2.waitKey(0)    
    key = cv2.waitKey(1)

    # ESC로 종료
    if key == 27 :
        break

cap.release()
cv2.destroyAllwindows()