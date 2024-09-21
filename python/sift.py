import cv2
import numpy as np
import base64
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

avg_center = None
z_depth = None
frame_base64 = None
lock = threading.Lock()

sift = cv2.SIFT_create(
    nfeatures=4000,
    nOctaveLayers=3,
    contrastThreshold=0.06,
    edgeThreshold=7,
    sigma=1.6
)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

logo_image = cv2.imread('python/pepsi_logo.jpg', cv2.IMREAD_GRAYSCALE)

if logo_image is None:
    print("Error loading logo image")
    exit()

h, w = 843, 632
area_original = h * w

keypoints_logo, descriptors_logo = sift.detectAndCompute(logo_image, None)

# Initialize Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

def encode_frame(frame):
    """Encode frame to JPEG format and then to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

@socketio.on('video_frame')
def handle_video_frame(data):
    global avg_center, z_depth, frame_base64

    # Decode the base64 frame
    frame_data = data['frame'].split(',')[1]
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame_data), np.uint8), cv2.IMREAD_COLOR)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray_frame, None)

    frame_base64 = encode_frame(frame)

    if des2 is not None:
        matches = flann.knnMatch(descriptors_logo, des2, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) > 40:
            src_pts = np.float32([keypoints_logo[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.5)

            if M is not None:
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                area_tracked = cv2.contourArea(np.int32(dst))

                if area_tracked > 0:
                    z_depth = area_original / area_tracked
                else:
                    z_depth = 0

                avg_center = np.mean(dst, axis=0).flatten()
                kalman.correct(np.array([[np.float32(avg_center[0])], [np.float32(avg_center[1])]]))
                prediction = kalman.predict()
                predicted_center = np.int32(prediction[:2].flatten())

                with lock:
                    socketio.emit('update_coordinates', {'x': float(avg_center[0]), 'y': float(avg_center[1]), 'z': float(z_depth)})
                    print(f"X: {float(avg_center[0]):.2f}, Y: {float(avg_center[1]):.2f}, Z: {float(z_depth):.2f}")
        else:
            with lock:
                socketio.emit('update_coordinates', {'x': None, 'y': None, 'z': None})

        with lock:
            socketio.emit('update_frame', {'frame': frame_base64})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
