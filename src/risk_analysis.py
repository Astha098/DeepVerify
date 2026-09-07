def calculate_risk(
    ela_score,
    copy_move_score,
    qr_detected,
    classification_confidence,
):
    risk = 0

    # ELA contribution
    if ela_score > 5:
        risk += 40
    elif ela_score > 2:
        risk += 20

    # Copy-move contribution
    if copy_move_score > 20:
        risk += 30

    # QR contribution
    if not qr_detected:
        risk += 10

    # Classification confidence
    if classification_confidence < 0.50:
        risk += 20

    return min(risk, 100)