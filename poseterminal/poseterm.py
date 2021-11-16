import cv2
import mediapipe as mp
import time

draw = True

draw_utils = mp.solutions.drawing_utils
poser = mp.solutions.pose

# populate reverse landmark dictionary
ldict={}
for i,x in enumerate(poser.PoseLandmark):
  ldict[i] = x.name

# Make dictionary of landmarks
# Set img_size to (width,height) tuple to scale coordinates to pixels
def landmarks2dict(landmarks,visibility_treshold=0.8,img_size=None):
   lms = {}
   if landmarks:
     for id, lm in enumerate(landmarks.landmark):
       if lm.visibility>visibility_treshold:
         if img_size:
           w,h = img_size
           lm.x, lm.y = lm.x * w, lm.y * h 
         lms[ldict[id]] = lm
   return lms

def draw_points(img,landmarks):
  d = landmarks2dict(landmarks,img_size=(img.shape[1],img.shape[0]))
  for k,v in d.items():
    cv2.circle(img, (int(v.x), int(v.y)), 5, (255, 0, 0), cv2.FILLED)

# TODO: Send coordinates to iot hub
def process(landmark_dict):
  pass


def main():
  cap = cv2.VideoCapture(0)
  pTime = 0
  detector = poser.Pose(
          static_image_mode=False,
          model_complexity=2, # 0,1,2
          enable_segmentation=True,
          smooth_segmentation=True,
          smooth_landmarks=True,
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5)

  while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.process(imgRGB)
    if draw and results.pose_landmarks:
        draw_utils.draw_landmarks(img, results.pose_landmarks, poser.POSE_CONNECTIONS)
        # draw_utils.plot_landmarks(img, results.pose_world_landmarks, poser.POSE_CONNECTIONS)        
        # draw_landmarks(img,results.pose_landmarks)
        d = landmarks2dict(results.pose_landmarks)
        process(d)
        # draw_points(img,results.pose_landmarks)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
      break


if __name__ == "__main__":
  main() 
