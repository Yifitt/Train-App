import cv2
import mediapipe as mp
import time
import math
import numpy as np
import playsound
import threading
from typing import Optional
from utilities import *
from timeit import default_timer as timer


class poseDetector():
    def __init__(self,mode = False,model_complexity =1,upBody = False,smooth = True,detCon = 0.5,trackCon = 0.5,static_image_mode=False,):
        self.mode = mode
        self.complex = model_complexity
        self.upBody = upBody
        self.smooth = smooth
        self.detCon = detCon
        self.trackCon = trackCon
        self.static_image_mode=  static_image_mode

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,self.mode,self.complex, self.upBody, self.smooth,
                                        self.detCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.results = None
        #Muscle up Variables
        self.dips_dir = 0
        self.pullup_dir = 0
        
        #Movement Variables
        self.state = "Waiting Command"
        self.count = 0
        self.dir = 0
        
        self.per1 = 0
        self.per2 = 0

        self.false_count = 0
        self.mybool = False


        self.bodypart = None
        self.previous_bodypart = None

        self.previous_per1,self.previous_per2 = 0,0

        self.warn = True

        # Static Movement Variable
        self.form = 0

        # Sound Variables
        self.good_form_AUDIO = r"C:\goruntu_isleme\projects\train\Perfectform_audio.mp3"
        self.go_upper_AUDIO = r"C:\goruntu_isleme\projects\train\Goupper.mp3"
        self.outro = r"C:\goruntu_isleme\kendi projelerim\outro.mp3"

        #Other
        self.times = nested_dict()
        self.ROM = nested_dict()
        self.workout = {}
        self.sets = 0

        #Rest Time
        self.rest_s = 15
        self.remaining = 0

        #Time
        self.total_time = 0
        self.end_time = 0
        self.start_time = 0



    def findPose(self,img,draw = True):
        
        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.imgRGB)
        if self.results != None:
            if self.results.pose_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
        
    def findPos(self,img,draw = True):
        self.lmList = []
        if img is None or not self.results.pose_landmarks:  
            return self.lmList
        for id,lm in enumerate(self.results.pose_landmarks.landmark):
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    self.lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
        return self.lmList
    
    def findAngle(self,img,p1,p2,p3,draw = True):
         
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]

        self.angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        if self.angle<0:
            self.angle+=360
        if self.angle>180:
            self.angle = 360-self.angle
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img,(x3,y3),(x2,y2),(255,255,255),3)
            cv2.circle(img,(x1,y1),5,(0,255,0),cv2.FILLED)
            cv2.circle(img,(x1,y1),15,(0,255,0),2)
            cv2.circle(img,(x2,y2),5,(0,255,0),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(0,255,0),2)
            cv2.circle(img,(x3,y3),5,(0,255,0),cv2.FILLED)
            cv2.circle(img,(x3,y3),15,(0,255,0),2)

        return self.angle
    
    def play_good_form(self):
        playsound.playsound(self.good_form_AUDIO)
    def play_go_upper(self):
        playsound.playsound(self.go_upper_AUDIO)

    def rest(self,start_time):
        if self.state == "Resting":
            self.remaining = counter(start_t=start_time,rest_period=self.rest_s)
            print(self.remaining)
            if self.remaining == 0:
                print(f"Rest Period is over.Waiting for command to stop!Time is ticking TICK TACK")
                self.state = "Waiting Command"
                self.remaining = 0
            return self.remaining

    def start_end_set(self,max):
        if max !="-":
            if self.count >=max:
                self.state="Resting"
                self.sets+=1
                self.false_count = 0
                self.workout[f"Set {self.sets}:"] = self.count
                self.count = 0
                print(f"Well Done! {self.sets}. set is completed!")
            if self.state == "END":
                clean_rom_dict = dictify(self.ROM)
                clean_time_dict = dictify(self.times)
                clean_time_dict = drop_rep1(clean_time_dict)
                #print(clean_time_dict)
                rom_abs = getROM(clean_rom_dict)
                #print(rom_abs)
                plotROM(rom_abs)
                plotTimes(clean_time_dict)
                

        return self.state
    
    def MovementMotion(self,max,min_v,max_v,bad_value,per = None,both = True,movement = "pull_up"):
        if both:
            if self.state == "beginning":
                if self.per1 <min_v and self.per2 <min_v:
                    self.state = "down"
                    self.ROM["Bench_Press"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]    
            elif self.state == "down":
                if self.per1 > max_v and self.per2 > max_v:
                    self.state = "up"
                    self.ROM[movement][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
            elif self.state == "up":
                if self.per1 < min_v and self.per2 < min_v:
                    self.state = "down"
                    self.ROM[movement][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
                    self.end_rep(min_v,True)
                    self.mybool = False
                elif self.per1>min_v and self.per2 > min_v:
                    self.detect_bad_from(bodypart=self.bodypart,bad_value = bad_value)
            if self.bodypart:
                self.previous_bodypart = self.bodypart
            self.previous_per1, self.previous_per2 = self.per1, self.per2
            self.state = self.start_end_set(max)
            return self.count,self.state,self.false_count,self.per1,self.per2
        else:
            if self.state == "beginning":
                if per <min_v:
                    self.state = "down"
                    self.ROM[movement][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]     
            elif self.state == "down":
                if per > max_v:
                    self.state = "up"
                    self.ROM[movement][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
            elif self.state == "up":
                if per< min_v:
                    self.state = "down"
                    self.ROM[movement][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
                    self.end_rep(min_v,both=False)
                    self.mybool = False
                elif per>min_v:
                    self.detect_bad_from(bad_value = bad_value,both=False)
                    #print(self.per1 > self.previous_per1+bad_value)
                    print(f"State:{self.state},Mybool:{self.mybool},Per1,Per2:{round(self.per1)},{round(self.per2)},PPer1,Pper2:{round(self.previous_per1)},{round(self.previous_per2)}")
            self.state = self.start_end_set(max)
            self.previous_per1= self.per1
            
            return self.count,self.state,self.false_count,self.per1
     
                       
    def detect_bad_from(self,bodypart = None,bad_value = 5,both = True):
        if both:
            if bodypart==None:
                if self.per1 > self.previous_per1+bad_value  or self.per2 > self.previous_per2+bad_value:
                        if self.warn:
                            threading.Thread(target=self.play_go_upper).start()
                            self.warn = False
                        self.mybool = True
            
            else:
                if self.previous_bodypart >self.bodypart+bad_value:
                    if self.warn:
                        threading.Thread(target=self.play_go_upper).start()
                        self.warn = False
                        self.mybool = True
                        self.previous_bodypart = self.bodypart        
        else:
            if self.previous_per1+bad_value < self.per1:
                if self.warn:
                    threading.Thread(target=self.play_go_upper).start()
                    self.warn = False
                self.mybool = True


    def end_rep(self,min_v,both):
        if both:
            if self.per1<=min_v and self.per2<=min_v:
                        if not self.mybool:
                            self.count+=1
                            if self.count in [2,4,6,10,15]:
                                threading.Thread(target=self.play_good_form).start()
                        elif self.mybool:
                            self.false_count+=1
                            self.warn = True
        else:
             if self.per1<=min_v:
                        if not self.mybool:
                            self.count+=1
                            if self.count in [2,4,6,10,15]:
                                threading.Thread(target=self.play_good_form).start()
                        elif self.mybool:
                            self.false_count+=1
                            self.warn = True


    def check_muscle_up(self,max,angle,angle2,*args):
        
        state = args[0] > args[2] and args[1] > args[3]

        if self.state =="beginning" and state:
                self.per1 = np.interp(angle,(50,150),(100,0))
                self.per2 = np.interp(angle2,(50,150),(100,0))
                if self.per1 <10 and self.per2 < 10:
                    self.state = "pull_up"  
                                                      
        if self.state == "pull_up" and self.pullup_dir==0 and state:
                self.per1 = np.interp(angle,(50,150),(100,0))
                self.per2 = np.interp(angle2,(50,150),(100,0))
                if self.per1 >80 or self.per2 > 80:
                        self.pullup_dir = 1
                        self.state= "dips"

        if self.state == "dips" and self.pullup_dir == 1 and  self.dips_dir == 0 and not state:
                self.per1  = np.interp(angle,(80,150),(100,0))
                self.per2 = np.interp(angle2,(80,150),(100,0))
                if self.per1 ==0 or self.per2 ==0: 
                    self.dips_dir = 1
                                     
        if self.dips_dir == 1 and self.state == "dips" and self.pullup_dir == 1 and not state:
            self.per1  = np.interp(angle,(80,150),(100,0))
            self.per2 = np.interp(angle2,(80,150),(100,0))
            if self.per1  >90 and self.per2 > 90:
                self.dips_dir = 0
                self.state = "pull_up" 

        if self.state == "pull_up" and self.pullup_dir==1 and self.dips_dir == 0 and state:
                self.per1 = np.interp(angle,(50,150),(100,0))
                self.per2 = np.interp(angle2,(50,150),(100,0))
                if self.per1 <10 and self.per2 <10:
                        self.pullup_dir = 0
                        self.count+=1
                        
        self.state =self.start_end_set(max)
        return self.count,self.state,0,self.per1,self.per2
    
    def check_bench_press(self,max,angle,angle2):
        self.per1 = np.interp(angle,(90,150),(100,0))
        self.per2 = np.interp(angle2,(90,150),(100,0))

        if self.state == "beginning":
            if self.per1 <25 or self.per2 <25:
                self.state = "down"
                self.ROM["Bench_Press"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]    
        elif self.state == "down":
            if self.per1 > 75 and self.per2 > 75:
                self.state = "up"
                self.ROM["Bench_Press"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
        elif self.state == "up":
            if self.per1 < 20 or self.per2 < 20:
                self.state = "down"
                self.ROM["Bench_Press"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
                self.count+=1
        self.state = self.start_end_set(max)        
        return self.count,self.state,self.false_count,self.per1,self.per2
    
    def check_deadlift(self,max,r_knee_angle,l_knee_angle,r_hip_angle, l_hip_angle):
        
        self.per1 = np.interp(r_hip_angle,(120,160),(100,0))
        self.per2 = np.interp(l_hip_angle,(120,160),(100,0))

        if self.state == "beginning":
            if self.per1 >90 or self.per2 >90:
                    if r_knee_angle > 60 and l_knee_angle > 60:
                            self.state = "up"
                            self.ROM["Deadlift"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 

        if self.state == "up":
                self.start_time = timer()
                if self.per1 < 10 and self.per2 < 10:
                    if r_knee_angle > 150 and l_knee_angle > 150:
                        if self.lmList[15][2] < self.lmList[25][2] and self.lmList[14][2] < self.lmList[26][2]:
                            self.count+=1
                            self.state="down"
                            self.ROM["Deadlift"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
        elif self.state=="down":
                if self.per1 > 90 or self.per2 > 90:
                    if r_knee_angle > 60 and l_knee_angle > 60:
                            self.state="up"
                            self.end_time = timer()
                            self.ROM["Deadlift"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                            self.total_time = round(self.end_time,2) - round(self.start_time,2)
                            self.times["Deadlift"][f"Set:{self.sets+1}"][f"Tempo_Rep:{self.count}:"] = self.total_time
        self.state = self.start_end_set(max)
        return self.count,self.state,0,self.per1,self.per2
    
    def check_shoulder_press(self,max,angle,angle2):
        self.per1 = np.interp(angle, (60, 170), (100, 0))
        self.per2 = np.interp(angle2, (60, 170), (100, 0))
        if self.state != "END":
            if self.state == "beginning":
                if self.per1 <10 and self.per2 <10:
                    self.state = "down"
                    self.ROM["Shoulder_Press"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]   
            elif self.state == "down":
                self.start_time = timer()
                if self.per1 > 90 and self.per2 > 90:
                    self.state = "up"
                    self.ROM["Shoulder_Press"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]   
            elif self.state == "up":
                if self.per1 < 10 and self.per2 < 10:
                    self.state = "down"
                    self.ROM["Shoulder_Press"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                    self.end_rep(10,True)
                    self.end_time = timer()
                    self.mybool = False
                    self.total_time = round(self.end_time,2) - round(self.start_time,2)
                elif self.per1>15 and self.per2 > 15:
                    self.detect_bad_from(bodypart=self.bodypart,bad_value = 20)
            self.times["Shoulder_Press"][f"Set:{self.sets+1}"][f"Tempo_Rep:{self.count}:"] = self.total_time
            
            self.previous_per1, self.previous_per2 = self.per1, self.per2 
        self.state= self.start_end_set(max)
        return self.count,self.state,self.false_count,self.per1,self.per2
            
       
    def check_deepsquat(self,max,angle,angle2):
        self.per1 = np.interp(angle,(20,170),(100,0))
        self.per2 = np.interp(angle2,(20,170),(100,0))
        
        return self.MovementMotion(10,90,15,True,max=max,movement="Deep_Squat")
    
    def check_dumble_curl(self,max,angle,angle2,both,which,*args): 
        stateB = [args[0]<args[2],args[1]<args[3]]

        self.per1 = np.interp(angle,(10,160),(100,0))
        self.per2 = np.interp(angle2,(10,160),(100,0))

        if both and stateB[0] and stateB[1]:  
           return self.MovementMotion(10,90,15,True,max=max,movement="Dumble_Curl")
        elif not both:
            if which == "right" and stateB[0] == True:
                return self.MovementMotion(10,90,15,self.per1,False,max=max,movement="Dumble_Curl")          
            elif which == "left"and stateB[1] == True:
                return self.MovementMotion(10,90,15,self.per2,False,max=max,movement="Dumble_Curl")    
        else:    
            return self.count,self.state,self.false_count,self.per1,self.per2
    
    def check_dips(self,max,angle,angle2):
        self.per1 = np.interp(angle,(80,150),(100,0))
        self.per2 = np.interp(angle2,(80,150),(100,0))       
        return self.MovementMotion(15,85,20,True,max=max,movement="Dips")
           
    def check_push_up(self,angle,angle2,max,*args):
        
        self.per1 = np.interp(angle,(95,150),(100,0))
        self.per2 = np.interp(angle2,(95,150),(100,0))
        
        state = args[0]< args[2] and args[1] < args[3]
        state1 = args[2] > args[4] and args[3] > args[5]

        if state == True and state1 ==True:
            if self.state == "beginning":
                if self.per1 <20 or self.per2 <20:
                    self.state = "down"
                    self.ROM["Pushup"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
            if self.per1 >75 and self.per2 >75:
                self.start_time = timer()
                if self.state == "down":
                    self.state ="up"
                    self.ROM["Pushup"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
            if self.per1 <20 or self.per2 <20:
                    if self.state == "up":
                        self.state = "down"
                        self.end_time = timer()
                        self.ROM["Pushup"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                        self.count+=1
                        self.total_time = round(self.end_time,2) - round(self.start_time,2)
            self.times["Pushup"][f"Set:{self.sets+1}"][f"Tempo_Rep:{self.count}:"] = self.total_time
        self.state = self.start_end_set(max)

        return self.count,self.state,0,self.per1,self.per2


    def check_pullup(self,max,angle,angle2,*args):
        self.per1 = np.interp(angle,(35,140),(100,0))
        self.per2 = np.interp(angle2,(35,140),(100,0))
        self.bodypart = args[5]

        return self.MovementMotion(10,85,35,True,max=max,movement="Pull_Up")
    
    def check_lsit(self,max,angle,angle2,r_knee_angle,l_knee_angle,r_hip_angle, l_hip_angle,fps):
        self.state = "lsit"
        if angle >150 and  angle2 >150 and r_knee_angle >150 and  l_knee_angle > 150 and  r_hip_angle >90 and  l_hip_angle >90:
            self.form+=1
            if self.form > fps:
                self.form-=fps
                self.count+=1
        self.state = self.start_end_set(max)           
        return self.count,self.state,0,0,0
    
    def check_plank(self,max,angle,angle2,fps,*args):
        state = args[0]< args[2] and args[1] < args[3]
        if state:
            self.state ="plank"
            if angle >70 and  angle2 >70:
                self.form+=1
                if self.form > fps:
                    self.form-=fps
                    self.count+=1
        else:
            self.state = "NO PLANK"

        self.state = self.start_end_set(max)   
        return self.count,self.state,0,angle,angle2  

    def check_seated_leg_extension(self,max,r_knee_angle,l_knee_angle):
        self.per1 = np.interp(r_knee_angle,(60,150),(100,0))
        self.per2 = np.interp(l_knee_angle,(60,150),(100,0))
        return self.MovementMotion(10,90,20,True,max=max,movement="Leg_Extension")
    
    def check_dumbel_lateral_raise(self,max,angle,angle2,both,which,armangr,armangl):    
        if both == True:
            self.per1 = np.interp(angle,(30,90),(100,0))
            self.per2 = np.interp(angle2,(30,90),(100,0))
            if armangr>150 and armangl>150:    
                if self.state == "beginning":
                    if self.per1  >95 and self.per2 > 95:
                        self.state = "up"
                        self.ROM["Lateral_Raise"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                if self.state == "up":
                    if self.per1 <10  and self.per2 <10:
                        self.state = "down"
                        self.ROM["Lateral_Raise"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                        self.end_rep(10,True)
                        self.mybool = False
                    elif self.per1>15 and self.per2 > 15:
                        self.detect_bad_from(bodypart=self.bodypart,bad_value = 15)   
                if self.state == "down":
                    if self.per1 > 95 and self.per2 >95:
                        self.state = "up"
                        self.ROM["Lateral_Raise"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
            self.previous_per1, self.previous_per2 = self.per1, self.per2               
        else:
            if which == "right":
                self.per1 = np.interp(angle,(30,90),(100,0))
                if armangr>150:    
                    if self.state == "beginning":
                        if self.per1  >95:
                            self.state = "up"
                            self.ROM["Lateral_Raise"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                    if self.state == "up":
                        if self.per1 <10:
                            self.state = "down"
                            self.ROM["Lateral_Raise"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                            self.end_rep(10,False)
                            self.mybool = False
                        elif self.per1>15:
                            self.detect_bad_from(bodypart=self.bodypart,bad_value = 15,both=False)   
                    if self.state == "down":
                        if self.per1 > 95:
                            self.state = "up"
                            self.ROM["Lateral_Raise"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                self.previous_per1 = self.per1
            elif which == "left":
                self.per1 = np.interp(angle2,(30,90),(100,0))
                if armangl>150:    
                    if self.state == "beginning":
                        if self.per1 > 95:
                            self.state = "up"
                            self.ROM["Lateral_Raise"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                    if self.state == "up":
                        if self.per1 <10:
                            self.state = "down"
                            self.ROM["Lateral_Raise"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                            self.end_rep(10,False)
                            self.mybool = False
                        elif self.per1 > 15:
                            self.detect_bad_from(bodypart=self.bodypart,bad_value = 15,both=False)   
                    if self.state == "down":
                        if self.per1 >95:
                            self.state = "up"
                            self.ROM["Lateral_Raise"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]
                self.previous_per1 = self.per1
        self.state = self.start_end_set(max)
        return self.count,self.state,self.false_count,self.per1,self.per2
    
    def check_dumbell_tricep_extension(self,max,angle,angle2,both,which):
        if both:
            self.per1 = np.interp(angle,(60,160),(100,0))
            self.per2 = np.interp(angle2,(60,100),(100,0))
            if self.state == "beginning":
                if self.per1 <10 and self.per2 <10:
                    self.state = "down" 
                    self.ROM["Tricep_Extension"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]    
            elif self.state == "down":
                if self.per1 > 90 and self.per2 > 90:
                    self.state = "up"
                    self.ROM["Tricep_Extension"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
            elif self.state == "up":
                if self.per1 < 10 and self.per2 < 10:
                    self.state = "down"
                    self.ROM["Tricep_Extension"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
                    self.end_rep(10,True)
                    self.mybool = False
                elif self.per1>10 and self.per2 > 10:
                    self.detect_bad_from(bodypart=self.bodypart,bad_value = 15)
    
            self.previous_per1, self.previous_per2 = self.per1, self.per2  
        else:
            if which == "right":
                self.per1 = np.interp(angle,(60,160),(100,0))
                if self.state == "beginning":
                    if self.per1 <10:
                        self.state = "down"
                        self.ROM["Tricep_Extension"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]     
                elif self.state == "down":
                    if self.per1 > 90:
                        self.state = "up"
                        self.ROM["Tricep_Extension"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
                elif self.state == "up":
                    if self.per1 < 10:
                        self.state = "down"
                        self.ROM["Tricep_Extension"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
                        self.end_rep(10,False)
                        self.mybool = False
                    elif self.per1>10:
                        self.detect_bad_from(bodypart=self.bodypart,bad_value = 15,both=False)
                self.previous_per1= self.per1
            if which == "left":
                self.per1 = np.interp(angle2,(60,160),(100,0))
                if self.state == "beginning":
                    if self.per1 <10:
                        self.state = "down"
                        self.ROM["Tricep_Extension"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2]    
                elif self.state == "down":
                    if self.per1 > 90:
                        self.state = "up"
                        self.ROM["Tricep_Extension"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
                elif self.state == "up":
                    if self.per1 < 10:
                        self.state = "down"
                        self.ROM["Tricep_Extension"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
                        self.end_rep(10,False)
                        self.mybool = False
                    elif self.per1 > 10:
                        self.detect_bad_from(bodypart=self.bodypart,bad_value = 15,both=False)
        
        self.state = self.start_end_set(max)        
        self.previous_per1 = self.per1
        return self.count,self.state,self.false_count,self.per1,self.per2
    
    def check_barbell_row(self,max,angle,angle2,r_knee_angle,l_knee_angle,hip1,hip2):
        self.per1 = np.interp(angle,(130,150),(100,0))
        self.per2 = np.interp(angle2,(130,150),(100,0))

        if (r_knee_angle < 160 or l_knee_angle < 160) and hip1 < 150 and hip2 < 150:
            if self.state == "beginning":
                if self.per1 >85 or self.per2 > 85:
                    self.state = "up"
                    self.ROM["Barbell_Row"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
            if self.per1 < 10  and self.per2 < 10:
                    if self.state == "down":
                        self.state = "up"
                        self.ROM["Barbell_Row"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 
                        self.count+=1
            if self.state == "up":
                if self.per1 >85 or self.per2 > 85:
                    self.state = "down"
                    self.ROM["Barbell_Row"][f"Set:{self.sets+1}"][f"State:{self.state}:"][f"Rep:{self.count+1}:"] = [self.per1,self.per2] 

        else:
            pass # buraya hip ve knee angle ayarlamasi gerektigini soyleyen json dondururuz kotlinde de o sesi oynatiriz brom
        
        self.state = self.start_end_set(max)
        return self.count,self.state,0,self.per1,self.per2
    
    def check_movement_type(self,img,movement,both = True,which = "right",max = 12,fps=0,lmlist = None,frame_width = None,frame_height = None):
        if movement == "muscle_up":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            shoulderR = lmlist[12][2]
            shoulderL = lmlist[11][2]
            thumbR = lmlist[22][2]
            thumbL = lmlist[21][2]
            return self.check_muscle_up(max,angle,angle2,shoulderR,shoulderL,thumbR,thumbL)
        if movement == "bench_press":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            return self.check_bench_press(max,angle,angle2)
        if movement == "deadlift":
            r_knee_angle = self.findAngle(img,24,26,28)
            l_knee_angle = self.findAngle(img,23,25,27)
            r_hip_angle = self.findAngle(img,12,24,26)
            l_hip_angle = self.findAngle(img,11,23,25)
            return self.check_deadlift(max,r_knee_angle,l_knee_angle,r_hip_angle, l_hip_angle)
        if movement == "deep_squat":
            angle = self.findAngle(img,24,26,28)
            angle2 = self.findAngle(img,23,25,27)
            return self.check_deepsquat(max,angle,angle2)
        if movement == "shoulder_press":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            return self.check_shoulder_press(max,angle,angle2)
        if movement == "dumble_curl":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            shoulderR = lmlist[12][2]
            shoulderL = lmlist[11][2]
            elbowL = lmlist[13][2]
            elbowR = lmlist[14][2]
            return self.check_dumble_curl(max,angle,angle2,both,which,shoulderR,shoulderL,elbowR,elbowL)
        if movement == "dips":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            return self.check_dips(max,angle,angle2)
        if movement == "pushup":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            shoulderR = lmlist[12][2]
            shoulderL = lmlist[11][2]
            thumbR = lmlist[22][2]
            thumbL = lmlist[21][2]
            kneeR = lmlist[26][2]
            kneeL = lmlist[25][2]
            return self.check_push_up(angle,angle2,max,shoulderR,shoulderL,thumbR,thumbL,kneeR,kneeL)
        if movement == "pullup":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            shoulderR = lmlist[12][2]
            shoulderL = lmlist[11][2]
            face = lmlist[10][2]
            wristR = lmlist[16][2]
            wristL = lmlist[15][2]
            valuedown = 10
            valueup = 85
            nose = lmlist[0][2]
            return self.check_pullup(max,angle,angle2,face,wristR,wristL,valuedown,valueup,nose)
        if movement == "lsit":
            r_knee_angle = self.findAngle(img,24,26,28)
            l_knee_angle = self.findAngle(img,23,25,27)
            r_hip_angle = self.findAngle(img,12,24,26)
            l_hip_angle = self.findAngle(img,11,23,25)
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            return self.check_lsit(max,angle,angle2,r_knee_angle,l_knee_angle,r_hip_angle, l_hip_angle,fps)
        if movement == "plank":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            shoulderR = lmlist[12][2]
            shoulderL = lmlist[11][2]
            kneeR = lmlist[26][2]
            kneeL = lmlist[25][2]
            return self.check_plank(max,angle,angle2,fps,shoulderR,shoulderL,kneeR,kneeL)
        if movement == "seated_legextension":
            r_knee_angle,l_knee_angle = self.findAngle(img,23,25,27),self.findAngle(img,24,26,28)
            return self.check_seated_leg_extension(max,r_knee_angle,l_knee_angle)
        if movement == "dumble_lateral_raise":
            angle = self.findAngle(img,24,12,14)
            angle2 = self.findAngle(img,23,11,13)
            armAngR = self.findAngle(img,12,14,16)
            armAngL = self.findAngle(img,11,13,15)
            return self.check_dumbel_lateral_raise(max,angle,angle2,both,which,armAngR,armAngL)
        if movement == "dumbell_tricep_extension":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            return self.check_dumbell_tricep_extension(max,angle,angle2,both,which)
        if movement == "barbell_row":
            angle = self.findAngle(img,12,14,16)
            angle2 = self.findAngle(img,11,13,15)
            hip1 = self.findAngle(img,12,24,26)
            hip2 = self.findAngle(img,11,23,25)
            r_knee_angle = self.findAngle(img,24,26,28)
            l_knee_angle = self.findAngle(img,23,25,27)
            return self.check_barbell_row(max,angle,angle2,r_knee_angle,l_knee_angle,hip1,hip2)
                                  
def main(): 
    per1 = 0
    per2 = 0
    false_count = 0
    pTime = 0
    reps = 0
    
    cap = cv2.VideoCapture(r"C:\goruntu_isleme\projects\train\videos\squatvid2.mp4")

    detector = poseDetector(detCon= 0.75,model_complexity=2,trackCon=0.5)
    
    movement = "deep_squat"
    state = None
    
    while True:
        ret,img = cap.read()
        #img = cv2.resize(img,(640,480))
        if ret:
            detector.findPose(img,True)
            lmList = detector.findPos(img,draw=False)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Genişlik
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Yükseklik
            
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            if lmList:   
                reps,state,false_count,per1,per2= detector.check_movement_type(img,movement,fps=fps,lmlist=lmList,frame_width=frame_width,frame_height=frame_height,both=False,which="right") 

            cv2.putText(img, f'FPS:{(int(fps))}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'Reps:{(int(reps))}', (40,150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'False Rep:{false_count}', (40,230), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'State:{state}', (40,310), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'PER1:{round(per1)}', (40,390), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'PER2:{round(per2)}', (40,470), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow("Frame",img)

        if cv2.waitKey(10) & 0xFF == ord('q'):                        
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
       main()