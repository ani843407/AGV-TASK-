import cv2
import numpy as np

# ---------- PARAMETERS ----------
FARNEBACK_PARAMS = dict(
    pyr_scale  = 0.5,   # pyramid scale
    levels     = 3,     # pyramid levels
    winsize    = 15,    # averaging window size
    iterations = 3,     # iterations per level
    poly_n     = 5,     # pixel neighbourhood size
    poly_sigma = 1.2,   # gaussian std for polynomial expansion
    flags      = 0,
)

FLOW_SCALE    = 3.0    # amplify small motions for visibility
BLEND_ALPHA   = 0.6    # overlay opacity

# ---------- MAIN ----------
cap = cv2.VideoCapture("video.mp4")

frame_idx = 0
while frame_idx < 940:
    ret, old_frame = cap.read(
    if not ret:
        print("Could not reach frame 940")
        cap.release()
        exit()
    frame_idx += 1

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# HSV canvas — hue encodes direction, value encodes magnitude
hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255   # full saturation always

while True:
    ret, frame = cap.read()
    if not ret or frame_idx > 1200:
        break
    frame_idx += 1

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- dense flow ----
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, **FARNEBACK_PARAMS)

    # flow[y, x] = (u, v) — convert to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2          # direction → hue (0-180)
    hsv[..., 2] = cv2.normalize(                  # magnitude → brightness
        mag * FLOW_SCALE, None, 0, 255, cv2.NORM_MINMAX
    )

    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # blend over original frame
    output = cv2.addWeighted(frame, 1 - BLEND_ALPHA, flow_bgr, BLEND_ALPHA, 0)

    # HUD
    mean_mag = float(np.mean(mag))
    cv2.putText(output, f"Frame: {frame_idx}  Mean flow: {mean_mag:.2f}px",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Dense Farneback Flow", output)
    old_gray = frame_gray.copy()

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()