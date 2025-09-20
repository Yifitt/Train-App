from HandDetection import handDetector
from Pose_Estimation import poseDetector
import cv2
import time

def main():
    movements =["shoulder_press","dumble_lateral_raise"]
    cap = cv2.VideoCapture(0)
    pose_detector = poseDetector(detCon= 0.75,model_complexity=2,trackCon=0.5)
    hand_detector = handDetector(poseDetector=pose_detector)
    reps = 0
    #per1 = 0
    #per2 = 0
    #false_count = 0
    pTime = 0
    remaining = 0
    #previous = 0
    gCommand = True
    start_t = None 
    i = 0
    if not gCommand:
        pose_detector.state = "beginning"

    while True:
        ret,img = cap.read()
    
        if ret:
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if pose_detector.state == "Waiting Command" or pose_detector.state == "Resting":
                pose_detector.state = hand_detector.finger_det(img)
            if pose_detector.state != "Waiting Command" and pose_detector.state != "Resting":
                pose_detector.findPose(img,True)
                lmList = pose_detector.findPos(img,draw=False)
                if lmList:
                    reps,pose_detector.state,false_count,per1,per2= pose_detector.check_movement_type(img,movements[i],fps=fps,max=3,lmlist=lmList,frame_width=frame_width,frame_height=frame_height,both=True,which="right")
            if pose_detector.state == "Next Movement":
                i+=1
                pose_detector.state = "beginning"

            cv2.putText(img, f'FPS:{(int(fps))}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'Reps:{(int(reps))}', (40,150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            if pose_detector.state == "Resting":
                 if start_t is None or start_t == 0:
                    start_t = time.time()
                 remaining = pose_detector.rest(start_t)
                 cv2.putText(img, f'Remaing:{remaining}', (40,310), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            else:
                start_t = None   
                remaining = 0
            #cv2.putText(img, f'False Rep:{false_count}', (40,230), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'State:{pose_detector.state}', (40,230), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            #cv2.putText(img, f'PER1:{round(per1)}', (40,390), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            #cv2.putText(img, f'PREVIOUS1:{round(per2)}', (40,470), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow("Frame",img)

        if (cv2.waitKey(10) & 0xFF == ord('q')) or pose_detector.state == "END":                        
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
       main()
       