import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import csv


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#resimAd açılar(8 tane açı) sınıf türü(altGogusUst vb)
acılar = []
#Açıyı hesaplamak için  fonk a b c -> ilk orta son
def calculate_angle(a,b,c):
    #Hesaplamayı kolaylaştırmak için np arrayine çeviriyoruz
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    # Belirlenen eklemler için radyanları hesaplayacak
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    # abs mutlak değere çevirici
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 




#resim sayacı
i=0
cap = cv2.VideoCapture("video/Bacak.mp4")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 1)
yazmaSayac=0
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        cap.set(cv2.CAP_PROP_FPS,1)
        ret,frame=cap.read()
        print(cap.get(cv2.CAP_PROP_FPS))

        frame = cv2.resize(frame,(800,800))

                
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
        # Make detection RGB
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            try:
                elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            except:
                print("elbowL bulunamadı")
                pass

            try:
                shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]  
            except:
                print("shoulderL bulunamadı")
                pass
            try:
                hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y] 
            except:
                print("hipL bulunamadı")
                pass
            try:
                elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            except:
                print("elbowR bulunamadı")

                pass

            try:
                shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            except:
                print("shoulderR bulunamadı")
                pass

            try:
                hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] 
            except:
                print("hipR bulunamadı")

                pass

    
            try:
                wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            except:
                print("wristL bulunamadı")

                pass
         
            try:
                wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            except:
                print("wristR bulunamadı")

                pass

            try:
                kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            except:
                print("kneeL bulunamadı")

                pass

            try:
                kneeR = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            except:
                print("kneeR bulunamadı")

                pass

            
            try:
                ankleL = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
               
            except:
                print("ankleL bulunamadı")

                pass
            try:
                ankleR = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            except:
                print("ankleR bulunamadı")

              
            try:

                shoulderLangle = calculate_angle(elbowL, shoulderL, hipL)
                shoulderRangle = calculate_angle(elbowR, shoulderR, hipR)
            except:
                print("shoulderRangle bulunamadı")
                pass

            try:

        
                elbowLangle = calculate_angle(shoulderL, elbowL, wristL)
                elbowRangle = calculate_angle(shoulderR, elbowR, wristR)

            
            except:
                print("nelbowRangle bulunamadı")
                pass

            try:
                hipLangle = calculate_angle(shoulderL, hipL, kneeL)
                hipRangle = calculate_angle(shoulderR, hipR, kneeR)
                
            except:
                print("nhipRangle bulunamadı")
                pass
            # Calculate angle

            try:

                kneeLangle = calculate_angle(hipL, kneeL, kneeL)
                kneeRangle = calculate_angle(hipR, kneeR, kneeR)
            
            except:
                print("nkneeRangle bulunamadı")

                pass 


        except:
            pass
       
                # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )     

        
        cv2.imshow("images[0]",image)
  
        
        acılar=[shoulderLangle,shoulderRangle,elbowLangle,elbowRangle,hipLangle,hipRangle,kneeLangle,kneeRangle]
        
        
        if yazmaSayac==3:
            row =acılar    
            row.insert(0,"Bacak")

            with open("kordinat.csv",mode="a",newline="") as f:
                    csv_writer = csv.writer(f,delimiter=",",quotechar='"',
                                                quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
            yazmaSayac =0
        
        acılar.clear
        yazmaSayac +=1


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


