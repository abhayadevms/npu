

import cv2
import numpy as np
import tensorflow as tf
import dlib
from tensorflow.keras.preprocessing.image import img_to_array


# Constants
AGE_RECOMMENDATIONS = {
    'Teen': 'Recommended product: Trendy gadgets',
    'Adult': 'Recommended product: Professional wear',
    'Senior': 'Recommended product: Health supplements'
}
GENDER_RECOMMENDATIONS = {
    'M': 'Recommended product: Latest tech gadgets',
    'F': 'Recommended product: Fashion accessories'
}



EMOTION_CLASSES = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def get_recommendation(age, gender):
    # Determine age group
    if age < 20:
        age_group = 'Teen'
    elif age < 60:
        age_group = 'Adult'
    else:
        age_group = 'Senior'

    # Get recommendations
    age_rec = AGE_RECOMMENDATIONS.get(age_group, None)
    gender_rec = GENDER_RECOMMENDATIONS.get(gender, 'No recommendation')
    return age_rec or gender_rec

# Load TensorFlow Lite model
# interpreter = tf.lite.Interpreter(model_path='models/model_age_gender.tflite')
# interpreter.allocate_tensors()

emotion_interpreter = tf.lite.Interpreter(model_path='models/emotion_model.tflite')
emotion_interpreter.allocate_tensors()

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Initialize Webcam
cap = cv2.VideoCapture('videos/supermarket.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output_video.mp4', fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))
img_size = 64

def preprocess_image(face):
    face = cv2.resize(face, (64, 64))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

while True:
    ret, frame = cap.read()
    if not ret:
        break
    preprocessed_faces_emo = []
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)
    detected = detector(frame, 1)
    faces = np.empty((len(detected), 64, 64, 3))

    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2 = d.left(), d.top(), d.right() + 1, d.bottom() + 1
            xw1 = max(int(x1 - 0.4 * d.width()), 0)
            yw1 = max(int(y1 - 0.4 * d.height()), 0)
            xw2 = min(int(x2 + 0.4 * d.width()), img_w - 1)
            yw2 = min(int(y2 + 0.4 * d.height()), img_h - 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            face = frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
            face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation=cv2.INTER_AREA)
            face_gray_emo = face_gray_emo.astype("float") / 255.0
            face_gray_emo = img_to_array(face_gray_emo)
            face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
            preprocessed_faces_emo.append(face_gray_emo)

        # # Make predictions with TensorFlow Lite model
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        #
        # # Convert faces to array and predict age and gender
        # faces_array = np.array(faces, dtype=np.float32)
        # interpreter.set_tensor(input_details[0]['index'], faces_array)
        # interpreter.invoke()
        # # print(interpreter.get_tensor(81))
        #
        # # Get predictions
        # predicted_gender = interpreter.get_tensor(output_details[0]['index'])
        # predicted_age = interpreter.get_tensor(output_details[1]['index'])
        # # print(predicted_gender)
        #
        # ages = np.arange(0, 101).reshape(101, 1)
        # predicted_ages = predicted_age.dot(ages).flatten()
        # print(predicted_gender[0][0])

        emotion_input_details = emotion_interpreter.get_input_details()
        emotion_output_details = emotion_interpreter.get_output_details()
        # print(emotion_output_details)

        emo_labels = []
        for i, d in enumerate(detected):

            emotion_interpreter.set_tensor(emotion_input_details[0]['index'], preprocessed_faces_emo[i])
            emotion_interpreter.invoke()
            emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])
            emo_labels.append(EMOTION_CLASSES[np.argmax(emotion_preds)])


        # # Draw results and add recommendations
        for i, d in enumerate(detected):
            # age = int(predicted_ages)
            # gender = "F" if predicted_gender[0][0] > 0.5 else "M"
            # recommendation = get_recommendation(age, gender)
            emotion = emo_labels[i]
            # label = "{}, {}, {}".format(age, gender, emotion)
            # draw_label(frame, (d.left(), d.top()), label + f", {recommendation}")
            # # Print recommendation to command line
            # print(f"Age: {age}, Gender: {gender}, Emotion:{emotion },  Recommendation: {recommendation}")

    cv2.imshow("Age and Gender Detector", frame)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
