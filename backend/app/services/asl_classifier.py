import numpy as np


def classify_asl(landmarks: list[list[float]]) -> str:
    """Rule-based ASL finger-spell classifier.

    Args:
        landmarks: List of 21 [x, y, z] hand landmark coordinates from MediaPipe.

    Returns:
        Single uppercase letter A-Z, or ' ' for space gesture.
    """
    if len(landmarks) != 21:
        return ""

    lm = np.array(landmarks)

    # --- Helper functions ---
    def is_finger_extended(tip: int, pip: int, mcp: int) -> bool:
        """Check if a finger is extended (tip above PIP and MCP)."""
        return lm[tip][1] < lm[pip][1] and lm[tip][1] < lm[mcp][1]

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(lm[a] - lm[b]))

    def tip_touching(a: int, b: int, threshold: float = 0.06) -> bool:
        return dist(a, b) < threshold

    # Finger states using tip, PIP, and MCP landmarks for robustness
    index_up = is_finger_extended(8, 6, 5)
    middle_up = is_finger_extended(12, 10, 9)
    ring_up = is_finger_extended(16, 14, 13)
    pinky_up = is_finger_extended(20, 18, 17)

    # Thumb: use x-axis (depends on hand orientation)
    # Compare thumb tip to thumb IP joint
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

    # --- Classification rules (ordered by specificity) ---
    # IMPORTANT: More specific checks must come before general catch-alls.
    # Within each finger-count group, check distinguishing features first.

    # === ONLY PINKY UP (count == 1, pinky) ===
    # Y: Thumb and pinky extended only
    if not index_up and not middle_up and not ring_up and pinky_up and thumb_out:
        return "Y"

    # I: Only pinky up, thumb in
    if not index_up and not middle_up and not ring_up and pinky_up and not thumb_out:
        return "I"

    # === ONLY INDEX UP (count == 1, index) ===
    # G: Index pointing sideways, thumb out (must check before L)
    if index_up and not middle_up and not ring_up and not pinky_up and thumb_out:
        index_dx = abs(lm[8][0] - lm[5][0])
        index_dy = abs(lm[8][1] - lm[5][1])
        if index_dx > index_dy:
            return "G"

    # L: Index up + thumb out, others down (L-shape, vertical index)
    if index_up and not middle_up and not ring_up and not pinky_up and thumb_out:
        return "L"

    # D: Index up only, thumb in
    if index_up and not middle_up and not ring_up and not pinky_up and not thumb_out:
        return "D"

    # === THREE FINGERS UP ===
    # F: Thumb + index touching, other three fingers up
    if not index_up and middle_up and ring_up and pinky_up and tip_touching(4, 8):
        return "F"

    # W: Index, middle, ring extended, pinky down
    if index_up and middle_up and ring_up and not pinky_up and not thumb_out:
        return "W"

    # === INDEX + MIDDLE UP (count == 2) ===
    # P: Similar to K but pointing down (check before R/V/K/H)
    if index_up and middle_up and not ring_up and not pinky_up:
        if lm[8][1] > lm[5][1]:  # Index pointing down
            return "P"

    # H: Index and middle pointing sideways (check before R/V)
    if index_up and middle_up and not ring_up and not pinky_up:
        index_dx = abs(lm[8][0] - lm[5][0])
        index_dy = abs(lm[8][1] - lm[5][1])
        if index_dx > index_dy:
            return "H"

    # R: Index and middle up and crossed (close together)
    if index_up and middle_up and not ring_up and not pinky_up:
        tip_dist = abs(lm[8][0] - lm[12][0])
        if tip_dist < 0.025:
            return "R"

    # V: Index and middle up and spread
    if index_up and middle_up and not ring_up and not pinky_up:
        tip_dist = abs(lm[8][0] - lm[12][0])
        if tip_dist > 0.04:
            return "V"

    # K: Index and middle up, thumb out
    if index_up and middle_up and not ring_up and not pinky_up and thumb_out:
        return "K"

    # U: Index and middle up and close together (default for 2 fingers up)
    if index_up and middle_up and not ring_up and not pinky_up:
        return "U"

    # === NO FINGERS UP, ONLY THUMB AREA ===
    # O: Thumb + index touching in circle (check before fist letters)
    if tip_touching(4, 8, 0.07) and not middle_up and not ring_up and not pinky_up:
        return "O"

    # Q: Index and thumb pointing down
    if not index_up and not middle_up and not ring_up and not pinky_up:
        if lm[8][1] > lm[5][1] and lm[4][1] > lm[3][1]:
            return "Q"

    # X: Index bent (hook shape)
    if not index_up and not middle_up and not ring_up and not pinky_up:
        if lm[7][1] < lm[6][1] and lm[8][1] > lm[7][1]:
            return "X"

    # === ALL FOUR FINGERS UP ===
    # B: All four fingers up, thumb across palm
    if all(fingers_up) and not thumb_out:
        return "B"

    # C: All fingers partially curved — thumb out, all up
    if all(fingers_up) and thumb_out:
        thumb_y = lm[4][1]
        index_base_y = lm[5][1]
        if thumb_y < index_base_y:
            return "C"
        return "B"

    # === FIST (count == 0) — ordered by specificity ===
    # T: Thumb between index and middle (thumb tip near index PIP)
    if count == 0 and tip_touching(4, 7, 0.05):
        return "T"

    # M: Thumb under index, middle, ring (all three knuckles over thumb)
    if count == 0 and lm[4][1] > lm[7][1] and lm[4][1] > lm[11][1]:
        return "M"

    # N: Thumb under index and middle only
    if count == 0 and lm[4][1] > lm[7][1] and lm[4][1] < lm[11][1]:
        return "N"

    # A: Fist with thumb to side
    if count == 0 and thumb_out:
        return "A"

    # S: Fist with thumb over fingers (thumb above middle MCP)
    if count == 0 and not thumb_out and thumb_tip[1] < lm[9][1]:
        return "S"

    # E: Fist with fingers curled, thumb under (catch-all for fist)
    if count == 0:
        return "E"

    # Default: space (unrecognized gesture)
    return " "
