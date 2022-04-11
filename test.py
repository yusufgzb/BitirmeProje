import cv2
print(cv2.__version__)
import numpy as np
from keras.models import model_from_json
from collections import Counter

import mediapipe as mp
import numpy as np

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img /255.0
    
    return img



model = model_from_json(open("model.json","r").read())
model.load_weights("gymWeights.h5")


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
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

cap = cv2.VideoCapture(0)



#hareket sayacılar
i=int()
say=[]
sayacArkaKol=0
sayacOmuz = 0
sayacÖnKolBar = 0
sayacOnKolCekic = 0
sayacAltGogus=0
sayacBacak=0
sayacSirt=0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while True:
        
        success, frame = cap.read()

        #  RGB       
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
    
        #  BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image =cv2.resize(image,(640,480))
        
        """
        Modelin tahmini için önce kordinatları bulma ve
        açıları hesaplama bölümü
        """
        try:
            landmarks = results.pose_landmarks.landmark

            try:
                elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            except:
                #print("elbowL bulunamadı")
                pass

            try:
                shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]  
            except:
                #print("shoulderL bulunamadı")
                pass
            try:
                hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y] 
            except:
                #print("hipL bulunamadı")
                pass
            try:
                elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                #print("elbowR bulunamadı")
            except:
                pass

            try:
                shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            
            except:
                #print("shoulderR bulunamadı")
                pass

            try:
                hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] 
            except:
                #print("hipR bulunamadı")

                pass

    
            try:
                wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            except:
                #print("wristL bulunamadı")

                pass
         
            try:
                wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            except:
                #print("wristR bulunamadı")

                pass

            try:
                kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            except:
                #print("kneeL bulunamadı")

                pass

            try:
                kneeR = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            
            except:
                #print("kneeR bulunamadı")

                pass

            
            try:
                ankleL = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
               
            except:
                #print("ankleL bulunamadı")

                pass
            try:
                ankleR = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            except:
                #print("ankleR bulunamadı")
                pass
              
            try:

                shoulderLangle = calculate_angle(elbowL, shoulderL, hipL)
                shoulderRangle = calculate_angle(elbowR, shoulderR, hipR)
                #print("shoulderRangle")
            except:
                #print("shoulderRangle")
                pass
            try:

                omuzGym = calculate_angle(elbowL, shoulderL, shoulderR)
                #print("shoulderRangle")
            except:
                #print("shoulderRangle")
                pass

            try:

        
                elbowLangle = calculate_angle(shoulderL, elbowL, wristL)
                elbowRangle = calculate_angle(shoulderR, elbowR, wristR)

            
            except:
                #print("nelbowRangle")
                pass

            try:
                hipLangle = calculate_angle(shoulderL, hipL, kneeL)
                hipRangle = calculate_angle(shoulderR, hipR, kneeR)

                
            
            except:
                #print("nhipRangle")
                pass
            # Calculate angle

            try:

                kneeLangle = calculate_angle(hipL, kneeL, kneeL)
                kneeRangle = calculate_angle(hipR, kneeR, kneeR)
            
            except:
                #print("nkneeRangle")

                pass 


        except:
            pass
       #aldığım 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )     
        acılar=np.array([shoulderLangle,shoulderRangle,elbowLangle,elbowRangle,hipLangle,hipRangle,kneeLangle,kneeRangle])
        say.append(acılar)
        if len(say)==8:
            x_test=np.array(say)

            say.clear()

            try:

                i=model.predict_classes(x_test)
                predToList=i.tolist()
                c = Counter(predToList)
                mostPered=c.most_common(1)
                mostPeredValue=mostPered[0][0]

                print(mostPeredValue)
            except:
                pass
            try:
                #0=AltGogus
                #1=Bacak
                #2=Sirt
                #3=arkaKol
                #4=omuz
                #5=ÖnKolBar
                if mostPeredValue==0:
                    cv2.putText(image, 'Flat Bench', (25, 50), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 2, cv2.LINE_AA)
                    if omuzGym > 170:
                        stage = "down"
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    if omuzGym < 125 and stage =='down':
                        stage="up"
                        sayacAltGogus +=1
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    cv2.putText(image, str(sayacAltGogus), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    

                elif mostPeredValue==1:

                    print('Bacak')
                    cv2.putText(image, 'Bacak', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    if hipLangle > 130:
                        stage = "down"
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    if hipLangle < 90 and stage =='down':
                        stage="up"
                        sayacBacak +=1
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    cv2.putText(image, str(sayacBacak), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    

                elif mostPeredValue==2:
                    cv2.putText(image, 'Lat Pull Down', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    if elbowRangle > 160:
                        stage = "down"
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    if elbowRangle < 60 and stage =='down':
                        stage="up"
                        sayacSirt +=1
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    cv2.putText(image, str(sayacSirt), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
                elif mostPeredValue==3:
                        
                    cv2.putText(image, 'Triceps Pushdown', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    if elbowRangle > 140:
                        stage = "down"
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    if elbowRangle < 120 and stage =='down':
                        stage="up"
                        sayacArkaKol +=1
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    cv2.putText(image, str(sayacArkaKol), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
                elif mostPeredValue==4:

                    cv2.putText(image, 'Side Raise', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    if shoulderLangle > 90:
                        stage = "up"
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    if shoulderLangle < 30 and stage =='up':
                        stage="down"
                        sayacOmuz +=1
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    cv2.putText(image, str(sayacOmuz), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
                    
                elif mostPeredValue==5:

                    cv2.putText(image, 'Biceps Curl', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    if elbowRangle > 150:
                        stage = "down"
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    if elbowRangle < 40 and stage =='down':
                        stage="up"
                        sayacÖnKolBar +=1
                        cv2.putText(image, str(stage), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    cv2.putText(image, str(sayacÖnKolBar), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                else:
                    print("Hareket algılanamadı")
        
            except:
                pass
        cv2.rectangle(image,(20,0),(590,65),(255,255,0),2)
        cv2.line(image,(320,0),(320,65),(255,255,0),2)
        cv2.line(image,(450,0),(450,65),(255,255,0),2)


        cv2.putText(image, "Hareketin Adi", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "Tekrar", (330, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "Konum", (460, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Proje",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()