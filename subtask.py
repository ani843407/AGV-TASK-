import cv2
import numpy as np

# ---------- PARAMETERS ----------
WIN_SIZE      = 7
MAX_ITER      = 10
LEVELS        = 3
MAX_TRACK_LEN = 15
MAX_CORNERS   = 200           # fewer, cleaner tracks
REDETECT_EVERY = 5
MIN_FEATURES  = 30

GFT_PARAMS = dict(
    maxCorners   = MAX_CORNERS,
    qualityLevel = 0.01,     # higher = more selective, less clutter
    minDistance  = 20,       # more spacing between tracks
    blockSize    = 7,
)

# Gradient: tail colour → head colour (BGR)
COLOR_TAIL = np.array([50,  50,  180], dtype=np.float32)   # dim red-ish
COLOR_HEAD = np.array([0,  230, 255], dtype=np.float32)    # bright cyan-yellow


# ---------- BUILD PYRAMID ----------
def build_pyramid(img, levels):
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid[::-1]          # coarse → fine


# ---------- LK FOR ONE POINT ----------
def lk_point(I1, I2, x, y):
    """Iterative LK at a single pyramid level. Returns (new_x, new_y)."""
    u, v = 0.0, 0.0
    h, w = I1.shape

    for _ in range(MAX_ITER):
        xi = int(round(x + u))
        yi = int(round(y + v))

        # boundary check on the *displaced* position
        if (xi - WIN_SIZE < 0 or yi - WIN_SIZE < 0 or
                xi + WIN_SIZE + 1 > w or yi + WIN_SIZE + 1 > h):
            return None          # out of bounds → drop this point

        I1_win = I1[y - WIN_SIZE : y + WIN_SIZE + 1,
                    x - WIN_SIZE : x + WIN_SIZE + 1].astype(np.float64)
        I2_win = I2[yi - WIN_SIZE : yi + WIN_SIZE + 1,
                    xi - WIN_SIZE : xi + WIN_SIZE + 1].astype(np.float64)

        if I1_win.shape != I2_win.shape:
            return None

        Ix = cv2.Sobel(I1_win, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(I1_win, cv2.CV_64F, 0, 1, ksize=3)
        It = I2_win - I1_win

        A   = np.vstack((Ix.flatten(), Iy.flatten())).T
        b   = -It.flatten()
        ATA = A.T @ A

        if np.linalg.det(ATA) < 1e-4:
            return None          # texture-less patch → drop

        nu       = np.linalg.inv(ATA) @ (A.T @ b)
        du, dv   = nu
        u       += du
        v       += dv

        if du * du + dv * dv < 0.01:
            break

    return x + u, y + v


# ---------- PYRAMIDAL LK ----------
def pyramidal_lk(I1, I2, points):
    """
    Track `points` from I1 → I2 using a Gaussian image pyramid.
    Returns list of (new_x, new_y) or None for lost points.
    """
    pyr1 = build_pyramid(I1, LEVELS)
    pyr2 = build_pyramid(I2, LEVELS)

    results = []

    for pt in points:
        x, y = float(pt[0][0]), float(pt[0][1])
        u, v = 0.0, 0.0

        result = None
        for lvl in range(LEVELS):
            scale   = 2 ** (LEVELS - lvl - 1)
            I1_lvl  = pyr1[lvl]
            I2_lvl  = pyr2[lvl]

            x_lvl = x / scale + u
            y_lvl = y / scale + v

            out = lk_point(I1_lvl, I2_lvl, int(round(x_lvl)), int(round(y_lvl)))
            if out is None:
                result = None
                break

            nx, ny = out
            u += nx - int(round(x_lvl))
            v += ny - int(round(y_lvl))

            # propagate to next (finer) level
            if lvl != LEVELS - 1:
                u *= 2
                v *= 2

            result = (x + u, y + v)

        results.append(result)

    return results


# ---------- MAIN ----------
cap = cv2.VideoCapture("video.mp4")

# ---- jump to frame 940 ----
frame_idx = 0
while frame_idx <940:
    ret, old_frame = cap.read()
    if not ret:
        print("Could not reach frame 940")
        cap.release()
        exit()
    frame_idx += 1

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

features = cv2.goodFeaturesToTrack(old_gray, **GFT_PARAMS)
# tracks: list of lists of (x, y) tuples — one list per active point
tracks = [[(float(f[0][0]), float(f[0][1]))] for f in features]

mask = np.zeros_like(old_frame)



# ---------- LOOP ----------
while True:
    mask = (mask * 0.93).astype(np.uint8)   # fade old trails
    ret, frame = cap.read()
    if not ret or frame_idx > 2000:
        break
    frame_idx += 1

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- track all active points ----
    if tracks:
        pts_in = np.array([[t[-1]] for t in tracks], dtype=np.float32)
        new_pts = pyramidal_lk(old_gray, frame_gray, pts_in)

        surviving_tracks = []

        for track, new_pt in zip(tracks, new_pts):
            if new_pt is None:
                continue

            nx, ny = new_pt
            x_prev, y_prev = track[-1]
            dx, dy = nx - x_prev, ny - y_prev
            dist2  = dx * dx + dy * dy

            if not (0.1 < dist2 < 5000):
                continue

            new_trail = track[-(MAX_TRACK_LEN - 1):] + [(nx, ny)]
            surviving_tracks.append(new_trail)

        tracks = surviving_tracks

    # ---- draw gradient tails ----
    for track in tracks:    
        if len(track) < 2:
            continue
        n = len(track)
        for i in range(1, n):
            t = i / (n - 1)                          # 0 = tail, 1 = head
            color = (COLOR_TAIL + t * (COLOR_HEAD - COLOR_TAIL)).astype(np.uint8)
            alpha = int(80 + 175 * t)                # fade tail, bright head
            p1 = (int(track[i-1][0]), int(track[i-1][1]))
            p2 = (int(track[i][0]),   int(track[i][1]))
            cv2.line(mask, p1, p2, color.tolist(), 2, cv2.LINE_AA)
        # bright dot at head
        cx, cy = int(track[-1][0]), int(track[-1][1])
        cv2.circle(mask, (cx, cy), 3, COLOR_HEAD.astype(np.uint8).tolist(), -1, cv2.LINE_AA)

    # ---- re-detect when tracks are sparse or on schedule ----
    if len(tracks) < MIN_FEATURES or frame_idx % REDETECT_EVERY == 0:
        new_features = cv2.goodFeaturesToTrack(frame_gray, **GFT_PARAMS)
        if new_features is not None:
            existing_pts = set(
                (int(round(t[-1][0] / 7)), int(round(t[-1][1] / 7)))
                for t in tracks
            )
            for f in new_features:
                fx, fy = float(f[0][0]), float(f[0][1])
                key = (int(round(fx / 7)), int(round(fy / 7)))
                if key not in existing_pts:
                    tracks.append([(fx, fy)])

    old_gray = frame_gray.copy()

    # ---- composite and display ----
    output = cv2.add(frame, mask)

    # HUD
    cv2.putText(output, f"Tracks: {len(tracks)}  Frame: {frame_idx}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Pyramidal LK – Improved", output)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()