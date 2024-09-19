import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load model architecture
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()  # Read the file content
json_file.close()

# Create model from JSON
model = model_from_json(model_json)

# Load weights into the model
model.load_weights("facialemotionmodel.h5")

# Load face cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize video input from a file
video_path = "D:\\opencv_projects\\face_emotions\\video.mp4"  # Replace with the path to your video file
video = cv2.VideoCapture(video_path)

# Get the original video's properties (frame width, height, frames per second)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save the output video in AVI format
output_path = "D:\\opencv_projects\\face_emotions\\output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = video.read()
    if not ret:
        print("End of video or failed to grab frame")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        cv2.putText(im, prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Write the frame with the rectangle and label to the output video
    out.write(im)

    cv2.imshow("Output", im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video input and output objects
video.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_path}")
