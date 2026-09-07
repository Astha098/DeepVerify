import cv2
import numpy as np


def detect_copy_move(path):
    image = cv2.imread(path)

    if image is None:
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(1000)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(descriptors, descriptors, k=2)

    suspicious_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        # Lowe's ratio test
        if m.distance < 0.75 * n.distance:

            pt1 = np.array(keypoints[m.queryIdx].pt)
            pt2 = np.array(keypoints[m.trainIdx].pt)

            distance = np.linalg.norm(pt1 - pt2)

            # Ignore identical points
            if distance > 20:
                suspicious_matches.append(m)
        output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

        cv2.imwrite("copy_move_result.jpg", output) 
    return len(suspicious_matches)