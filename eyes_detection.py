import os
import urllib.request
import cv2
import mediapipe as mp

# Official off-the-shelf model used in MediaPipe samples (short-range). :contentReference[oaicite:3]{index=3}
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = "blaze_face_short_range.tflite"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def detect_eye_centers_mediapipe(image_bgr):
    ensure_model()

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE
    )

    # MediaPipe Tasks wants mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with FaceDetector.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.detections:
        return None

    # pick the largest face
    def area(det):
        bb = det.bounding_box
        return bb.width * bb.height

    det = max(result.detections, key=area)

    # Face Detector outputs keypoints including left_eye and right_eye. :contentReference[oaicite:4]{index=4}
    # Keypoints are normalized coords in [0,1] relative to image.
    kps = det.keypoints

    left_eye = kps[0]   # typically left_eye
    right_eye = kps[1]  # typically right_eye

    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)

    return (lx, ly), (rx, ry)

if __name__ == "__main__":
    # pip install mediapipe opencv-python
    img = cv2.imread("face.png")
    out = detect_eye_centers_mediapipe(img)
    if out is None:
        print("No face found.")
        raise SystemExit

    (lx, ly), (rx, ry) = out
    print("Left eye :", (lx, ly))
    print("Right eye:", (rx, ry))

    vis = img.copy()
    cv2.circle(vis, (lx, ly), 4, (0, 255, 0), -1)
    cv2.circle(vis, (rx, ry), 4, (0, 255, 0), -1)
    cv2.imshow("eyes", vis)
    cv2.waitKey(0)
