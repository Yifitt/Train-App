import cv2
import mediapipe as mp
from typing import Optional

class handDetector():
    def __init__(self,poseDetector=None, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5,static_image_mode=False):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.static_image_mode=  static_image_mode
        
        self.lmList = []

        self.poseDetector = poseDetector

        self.fingers = None
        self.finger_count = 0
        self.flag = 0
        
    def findHands(self, img, draw=True):
        if img is None:  
            return None

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_img)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img  

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if img is None or not self.results.multi_hand_landmarks:  
            return self.lmList
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        return self.lmList
      
    def findFingers(self):
        self.fingers = [0,0,0,0,0]
        self.finger_count = 0

        #Thumb
        if self.lmList:
            if self.lmList[4][1] < self.lmList[2][1]:     
               if self.fingers[0] == 0:
                    self.finger_count+=1
                    self.fingers[0] = 1
            else:
                self.fingers[0] = 0
            #Index
            if self.lmList[8][2] < self.lmList[6][2]:   
                if self.fingers[1] == 0:
                    self.finger_count+=1
                    self.fingers[1] = 1
            else:
                self.fingers[1] = 0
            #Middle
            if self.lmList[12][2] < self.lmList[10][2]:      
                if self.fingers[2] == 0:
                    self.finger_count+=1
                    self.fingers[2] = 1
            else:
                self.fingers[2] = 0
            #Ring
            if self.lmList[16][2] < self.lmList[14][2]:
                if self.fingers[3] == 0:
                    self.finger_count+=1
                    self.fingers[3] = 1
            else:
                self.fingers[3] = 0
            #Pinky
            if self.lmList[20][2] < self.lmList[18][2]:
                if self.fingers[4] == 0:
                    self.finger_count+=1
                    self.fingers[4] = 1
            else:
                self.fingers[4] = 0
            
        return self.finger_count,self.fingers
    
    def finger_det(self,image):
        if self.poseDetector.state == "Waiting Command":
            image = self.findHands(image)
            self.lmList = self.findPosition(image)
            if self.lmList:
                self.findFingers()
                if self.fingers:
                    if self.fingers[0] == 1 and self.fingers[1] == 1 and self.fingers[2] == 0 and self.fingers[3] == 0 and self.fingers[4] == 0 and self.finger_count == 2:
                        self.flag+=1
                        if self.flag>15:
                            self.poseDetector.state = "beginning"
                            self.flag = 0
                    elif self.fingers[0] == 1 and self.fingers[1] == 1 and self.fingers[2] == 0 and self.fingers[3] == 0 and self.fingers[4] == 1 and self.finger_count == 3:
                        self.flag+=1
                        if self.flag>15:
                            self.poseDetector.state = "END"
                            self.flag = 0
                    elif self.fingers[0] == 1 and self.fingers[1] == 1 and self.fingers[2] == 1 and self.fingers[3] == 1 and self.fingers[4] == 0 and self.finger_count == 4:
                        self.flag+=1
                        if self.flag>15:
                            self.poseDetector.state = "Next Movement"
                            self.flag = 0
        return self.poseDetector.state



