import cv2
import numpy as np
import face_recognition

def test(image, model_dir=None, device_id=0):
    """
    Simulated anti-spoofing detector.
    Uses sharpness and presence of face to determine real vs spoof.

    Args:
        image (numpy.ndarray): BGR webcam frame
        model_dir (str): [Optional] Path to ML models
        device_id (int): [Optional] GPU device ID (not used here)

    Returns:
        int: 1 = Real face, 0 = Spoof or unclear input
    """

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Check image sharpness (simulates spoof check)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Step 3: Ensure a real face exists in the frame
    face_locations = face_recognition.face_locations(image)
    face_detected = len(face_locations) > 0

    # Thresholds (tweak if needed)
    sharpness_threshold = 100  # lower means blurred
    if face_detected and laplacian_var > sharpness_threshold:
        return 1  # Real face detected
    else:
        return 0  # Spoof, unclear, or no face
