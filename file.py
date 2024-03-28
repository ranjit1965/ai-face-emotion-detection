from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model("./model.h5")

class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Load the image you want to predict
image_path = "./suprise.jpeg"
frame = cv2.imread(image_path)
#cv2.imshow("image",frame)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray)

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        print("\nprediction = ", preds)
        label = class_labels[preds.argmax()]
        print("\nprediction max = ", preds.argmax())
        print("\nlabel = ", label)
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("\n\n")

cv2.imshow('Emotion Detector', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
