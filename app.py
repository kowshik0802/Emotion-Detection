from flask import Flask, render_template, Response, request, jsonify
import cv2
from collections import Counter
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import sys
import os
import tensorflow as tf

# Import chatbot script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
if float(tf.__version__[0]) >= 2.0:
    from tf2 import chatbot_tf2 as chbot
else:
    from tf1 import chatbot_tf1 as chbot

# Define hyperparameters using argparse
parser = argparse.ArgumentParser(description='GPT-2 chatbot')
parser.add_argument('--nsamples', type=int, default=1,
                    help='set number of bot outputs')
parser.add_argument('--top_k', type=int, default=5,
                    help='set limited to only number of k words in order of highest probability')
parser.add_argument('--top_p', type=float, default=1.0,  # Changed type to float
                    help='set sum probability p that only words exceeding p are put in the candidate')
parser.add_argument('--temperature', type=float, default=0.6,
                    help='write flexibly if the temperature is high, and write statically if the temperature is low (0.0 ~ 1.0)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='set the batch size')
parser.add_argument('--length', type=int, default=20,
                    help='set the response maximum number of length')

args = parser.parse_args()

# Initialize Flask app
app = Flask(__name__)

# Initialize global variables for emotion detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('Emotion_Detection.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_counter = Counter()
capture_count = 0
final_emotion = None

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

# Route for chatbot endpoint
@app.route('/chatbot', methods=['POST'])
# Modify the generate_chatbot_response function to accept the user's message
# Function to generate response from the chatbot
# Function to generate response from the chatbot
def generate_chatbot_response(user_message):
    # Generate response using chatbot logic
    response = chbot.interact_model(
        nsamples=args.nsamples,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        batch_size=args.batch_size,
        length=args.length,
        user_message=user_message  # Pass the user message as an argument
    )
    return response
# Modify the chatbot route to handle the chatbot response correctly
# Route for chatbot endpoint
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Extract user message from the request JSON
    user_message = request.json.get('message')
    
    # Generate response from the chatbot
    response_message = generate_chatbot_response(user_message)
    
    # Return response as JSON
    return jsonify({'message': response_message[0]})  # Ensure to return the response message as JSON





# Route for emotion detection using face detection
@app.route('/emotion_detection')
def emotion_detection():
    return Response(emotion_detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function for emotion detection
def emotion_detect():
    global emotion_counter, final_emotion, capture_count

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Make a prediction on the ROI, then lookup the class
                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]

                # Increment the counter for the detected emotion
                emotion_counter[label] += 1

                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Display the frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Check if the most common emotion has been detected 10 times
        if capture_count == 16:
            most_common_emotion = emotion_counter.most_common(1)
            
            if most_common_emotion and most_common_emotion[0][1] == 16:
                final_emotion = most_common_emotion[0][0]
                print("Emotion detected 10 times:", final_emotion)
                break
            
            # Reset capture count and emotion counter for the next ten captures
            capture_count = 0
            emotion_counter.clear()
        
        # Increment the capture count
        capture_count += 1
        
        # Quit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)
