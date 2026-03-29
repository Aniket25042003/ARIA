import numpy as np


def classify_asl(landmarks: list[list[float]]) -> tuple[str, float]:
    """Rule-based ASL finger-spell classifier with confidence scoring.

    Args:
        landmarks: List of 21 [x, y, z] hand landmark coordinates from MediaPipe.

    Returns:
        Tuple of (letter, confidence) where letter is A-Z or ' ' and
        confidence is 0.0-1.0 indicating classification certainty.
    """
    if len(landmarks) != 21:
        return "", 0.0

    lm = np.array(landmarks)

    # --- Helper functions ---
    def is_finger_extended(tip: int, pip: int, mcp: int) -> bool:
        """Check if a finger is extended (tip above PIP and MCP in Y-axis)."""
        return lm[tip][1] < lm[pip][1] and lm[tip][1] < lm[mcp][1]

    def is_finger_curled(tip: int, pip: int, mcp: int) -> bool:
        """Check if a finger is curled (tip below PIP)."""
        return lm[tip][1] > lm[pip][1]

    def dist(a: int, b: int) -> float:
        """3D Euclidean distance between two landmarks."""
        return float(np.linalg.norm(lm[a] - lm[b]))

    def dist_2d(a: int, b: int) -> float:
        """2D distance (ignoring depth) between two landmarks."""
        return float(np.linalg.norm(lm[a][:2] - lm[b][:2]))

    def tip_touching(a: int, b: int, threshold: float = 0.06) -> bool:
        return dist(a, b) < threshold

    def finger_angle(tip: int, pip: int, mcp: int) -> float:
        """Angle at PIP joint in degrees — 180 = fully extended, <90 = curled."""
        v1 = lm[tip] - lm[pip]
        v2 = lm[mcp] - lm[pip]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    def hand_span() -> float:
        """Distance from wrist to middle fingertip — used to normalize thresholds."""
        return dist(0, 12)

    # --- Compute finger states ---
    index_up = is_finger_extended(8, 6, 5)
    middle_up = is_finger_extended(12, 10, 9)
    ring_up = is_finger_extended(16, 14, 13)
    pinky_up = is_finger_extended(20, 18, 17)

    index_curled = is_finger_curled(8, 6, 5)
    middle_curled = is_finger_curled(12, 10, 9)
    ring_curled = is_finger_curled(16, 14, 13)
    pinky_curled = is_finger_curled(20, 18, 17)

    # Thumb: use x-axis direction relative to hand orientation
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]
    wrist = lm[0]

    # Determine handedness by comparing middle MCP to wrist x
    is_right_hand = lm[9][0] > wrist[0]
    if is_right_hand:
        thumb_out = thumb_tip[0] > thumb_ip[0]
    else:
        thumb_out = thumb_tip[0] < thumb_ip[0]

    thumb_up = thumb_tip[1] < thumb_ip[1] and thumb_tip[1] < thumb_mcp[1]

    fingers_up = [index_up, middle_up, ring_up, pinky_up]
    count = sum(fingers_up)

    # Adaptive threshold based on hand size
    span = hand_span()
    touch_threshold = span * 0.15  # ~6% of hand span for touching
    close_threshold = span * 0.08  # Fingers "close together"
    spread_threshold = span * 0.12  # Fingers "spread apart"

    # Finger angles for more robust extension detection
    index_angle = finger_angle(8, 6, 5)
    middle_angle = finger_angle(12, 10, 9)

    # --- Classification rules (ordered by specificity) ---

    # === ONLY PINKY UP (count == 1, pinky) ===
    if not index_up and not middle_up and not ring_up and pinky_up:
        if thumb_out:
            return "Y", 0.90
        return "I", 0.85

    # === ONLY INDEX UP (count == 1, index) ===
    if index_up and not middle_up and not ring_up and not pinky_up:
        index_dx = abs(lm[8][0] - lm[5][0])
        index_dy = abs(lm[8][1] - lm[5][1])

        # G: Index pointing sideways with thumb out
        if thumb_out and index_dx > index_dy:
            return "G", 0.85

        # L: Index up + thumb out (L-shape, vertical index)
        if thumb_out:
            return "L", 0.90

        # D: Index up only, thumb in
        return "D", 0.85

    # === THREE FINGERS UP (count == 3) ===
    if count == 3:
        # F: Thumb + index form a circle, other three fingers up
        if not index_up and middle_up and ring_up and pinky_up and tip_touching(4, 8, touch_threshold):
            return "F", 0.85

        # W: Index, middle, ring extended, pinky down, thumb in
        if index_up and middle_up and ring_up and not pinky_up:
            if not thumb_out:
                return "W", 0.90
            # 3 fingers up + thumb out could be a sloppy W
            return "W", 0.65

    # === INDEX + MIDDLE UP (count == 2) ===
    if index_up and middle_up and not ring_up and not pinky_up:
        index_dx = abs(lm[8][0] - lm[5][0])
        index_dy = abs(lm[8][1] - lm[5][1])

        # P: Index and middle pointing downward
        if lm[8][1] > lm[5][1] and lm[12][1] > lm[9][1]:
            return "P", 0.80

        # H: Index and middle pointing sideways
        if index_dx > index_dy:
            return "H", 0.80

        # Now check vertical configurations
        tip_dist_2d = dist_2d(8, 12)
        tip_dist_x = abs(lm[8][0] - lm[12][0])

        # R: Index and middle crossed (very close tips, and middle crosses over index)
        if tip_dist_x < close_threshold and lm[12][2] != lm[8][2]:
            # Use depth (Z) to detect crossing — crossed fingers have different Z values
            z_diff = abs(lm[8][2] - lm[12][2])
            if z_diff > 0.01 or tip_dist_x < close_threshold * 0.7:
                return "R", 0.75

        # V: Index and middle spread apart
        if tip_dist_2d > spread_threshold:
            return "V", 0.90

        # K: Index and middle up with thumb out between them
        if thumb_out and thumb_tip[1] < lm[9][1]:
            return "K", 0.80

        # U: Index and middle up and close together (default 2-finger)
        return "U", 0.75

    # === THUMB + INDEX TOUCHING (before fist checks) ===
    if tip_touching(4, 8, touch_threshold) and not middle_up and not ring_up and not pinky_up:
        # O: Circle shape — thumb and index tips touching
        return "O", 0.80

    # === NO FINGERS UP, POINTING DOWN ===
    if count == 0:
        # Q: Index and thumb both pointing down
        if lm[8][1] > lm[5][1] and lm[4][1] > lm[3][1]:
            index_below = lm[8][1] - lm[5][1]
            thumb_below = lm[4][1] - lm[3][1]
            if index_below > 0.02 and thumb_below > 0.02:
                return "Q", 0.70

        # X: Index bent in hook shape (DIP above PIP, but TIP below DIP)
        if lm[7][1] < lm[6][1] and lm[8][1] > lm[7][1]:
            return "X", 0.75

    # === ALL FOUR FINGERS UP (count == 4) ===
    if all(fingers_up):
        if not thumb_out:
            return "B", 0.90
        # C: All up with thumb out and curved
        thumb_y = lm[4][1]
        index_base_y = lm[5][1]
        if thumb_y < index_base_y:
            return "C", 0.80
        return "B", 0.75

    # === FIST VARIANTS (count == 0) — ordered by specificity ===
    if count == 0:
        # T: Thumb tip between index and middle (near index PIP)
        if tip_touching(4, 7, touch_threshold * 0.8):
            return "T", 0.75

        # M: Thumb under index, middle, and ring
        thumb_under_index = lm[4][1] > lm[7][1]
        thumb_under_middle = lm[4][1] > lm[11][1]
        thumb_under_ring = lm[4][1] > lm[15][1]

        if thumb_under_index and thumb_under_middle and thumb_under_ring:
            return "M", 0.70

        # N: Thumb under index and middle only (not ring)
        if thumb_under_index and thumb_under_middle and not thumb_under_ring:
            return "N", 0.70

        # A: Fist with thumb to the side
        if thumb_out:
            return "A", 0.85

        # S: Fist with thumb over fingers (thumb above middle MCP)
        if not thumb_out and thumb_tip[1] < lm[9][1]:
            return "S", 0.75

        # E: Fist catch-all
        return "E", 0.65

    # Default: unrecognized gesture
    return " ", 0.30
