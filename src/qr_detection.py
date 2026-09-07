import cv2


def detect_qr(path):
    image = cv2.imread(path)

    rotations = [
        image,
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    detector = cv2.QRCodeDetector()

    for img in rotations:
        data, bbox, _ = detector.detectAndDecode(img)

        if data:
            return data

    return "QR code not detected"