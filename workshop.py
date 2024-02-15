from flask import Flask, render_template, Response
import cv2
import winsound
import time
import os

app = Flask(__name__)

# Ensure that the "images" directory exists
if not os.path.exists("images"):
    os.makedirs("images")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print('not succeed')
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Increase the height (h) to include the hair and shoulders
                extended_h = int(1.5 * h)

                # Calculate the new y-coordinate
                new_y = max(0, y - (extended_h - h) // 2)

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, new_y), (x + w, new_y + extended_h), (0, 255, 0), 2)

                # Get the current time and format it
                current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                message = f"Face detected at {current_time}"

                # Print the message to the terminal
                print(message)

                # Save the image with the detected face (including hair and shoulders) in the "images" folder
                filename = os.path.join("images", f"full_face_{current_time}.jpg")
                cv2.imwrite(filename, frame[new_y:new_y+extended_h, x:x+w])

                # Play a beep sound (adjust the frequency and duration as needed)
                winsound.Beep(1000, 500)  # Example: 1000 Hz for 500 ms

            ret, buffer = cv2.imencode('.jpg', frame)

            if not ret:
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
