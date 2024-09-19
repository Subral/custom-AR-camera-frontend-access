import cv2
import numpy as np
import base64
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS  # Import CORS
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins

# Global variables to store object position, depth, and video feed
avg_center = None
z_depth = None
frame_base64 = None
lock = threading.Lock()

# Initialize the SIFT detector with custom attributes
sift = cv2.SIFT_create(
    nfeatures=3000,         # Number of best features to retain
    nOctaveLayers=3,        # Number of layers in each octave
    contrastThreshold=0.06, # Threshold for contrast filter
    edgeThreshold=7,       # Threshold for edge detection
    sigma=1.6               # Gaussian filter standard deviation
)


# FLANN parameters for SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # Higher value means better precision, but slower
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load the Pepsi logo (query image)
logo_image = cv2.imread('python/pepsi_logo.jpg', cv2.IMREAD_GRAYSCALE)

if logo_image is None:  
    print("Error loading logo image")
    exit()

# The original image size is 842 x 843, so the area is:
h, w = 843, 632  # Logo image dimensions
area_original = h * w  # Area of the original logo image

# Detect keypoints and descriptors in the logo image using SIFT
keypoints_logo, descriptors_logo = sift.detectAndCompute(logo_image, None)

cap = cv2.VideoCapture(0)

def encode_frame(frame):
    """Encode frame to JPEG format and then to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def track_object():
    global avg_center, z_depth, frame_base64

    past_boxes = []
    N = 5

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect keypoints and descriptors in the live frame using SIFT
        kp2, des2 = sift.detectAndCompute(gray_frame, None)

        # Always convert the current frame to base64 and send it
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

                    past_boxes.append(dst)
                    if len(past_boxes) > N:
                        past_boxes.pop(0)

                    avg_box = np.mean(past_boxes, axis=0)

                    # Calculate the area of the tracked object
                    area_tracked = cv2.contourArea(np.int32(dst))

                    # Calculate Z-depth: ratio of original to tracked area
                    if area_tracked > 0:
                        z_depth = area_original / area_tracked
                    else:
                        z_depth = 0

                    avg_center = np.mean(avg_box, axis=0).flatten()
                    kalman.correct(np.array([[np.float32(avg_center[0])], [np.float32(avg_center[1])]]))
                    prediction = kalman.predict()
                    predicted_center = np.int32(prediction[:2].flatten())

                    # Send cube coordinates
                    with lock:
                        socketio.emit('update_coordinates', {'x': float(avg_center[0]), 'y': float(avg_center[1]), 'z': float(z_depth)})
                        print(f"X: {float(avg_center[0]):.2f}, Y: {float(avg_center[1]):.2f}, Z: {float(z_depth):.2f}")
            else:
                # Reset the coordinates when no marker is detected
                with lock:
                    socketio.emit('update_coordinates', {'x': None, 'y': None, 'z': None})

        # Always send the frame regardless of marker detection
        with lock:
            socketio.emit('update_frame', {'frame': frame_base64})

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Start object tracking in a separate thread
    tracking_thread = threading.Thread(target=track_object)
    tracking_thread.start()

    # Start the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000)
