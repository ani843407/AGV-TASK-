"""
controller.py
=============
Optical-Flow Navigation  —  Lucas-Kanade method

State machine:
  CRUISE      → driving straight down centre (target y=0)
  APPROACH    → obstacle within APPROACH_X, begin steering to clear side
  AVOID       → actively steering around obstacle
  STRAIGHTEN  → obstacle passed, steering back to y=0

Run:
    xvfb-run -a python3 controller.py
"""

from pyvirtualdisplay import Display
_display = Display(visible=0, size=(1280, 720))
_display.start()

import os, time
import cv2
import numpy as np
import pybullet as p
from simulation_setup import setup_simulation

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
CAM_W, CAM_H = 640, 480
CAM_FOV      = 70
CAM_NEAR     = 0.1
CAM_FAR      = 25.0
FPS          = 30
DT           = 1.0 / 60.0

GOAL_X       = 28.0
BASE_SPEED   = 14.0
AVOID_SPEED  = 9.0
MAX_STEER    = 0.5
STEER_GAIN   = 0.90

ROAD_HALF_W  = 1.16
BOUNDARY_SAFE= 0.40

# Obstacle positions from simulation_setup.py
OBS_POSITIONS = [
    (6,  +0.38),
    (12, -0.38),
    (18, +0.38),
    (24, -0.38),
    (30, +0.38),
]

# Clear-side Y target for each obstacle (opposite side, 60% of road half-width)
# Obs at +0.38 → go to y = -0.60 (left gap is wider)
# Obs at -0.38 → go to y = +0.60 (right gap is wider)
CLEAR_Y = {ox: -np.sign(oy) * 0.60 for ox, oy in OBS_POSITIONS}

APPROACH_X  = 4.0    # start APPROACH state this far before obstacle
CLEARED_X   = 1.8    # switch to STRAIGHTEN this far past obstacle
STRAIGHT_TOL= 0.08   # y error below this → back to CRUISE

LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FEAT_PARAMS = dict(maxCorners=200, qualityLevel=0.005, minDistance=6, blockSize=7)
VIDEO_PATH  = "/tmp/sim_video.avi"

# ═══════════════════════════════════════════════════════════════════════
# SIMULATION INIT
# ═══════════════════════════════════════════════════════════════════════
car_id, steering_joints, motor_joints = setup_simulation(
    dt=DT, settle_frames=60, gui=True)
print(f"\n[Init] car_id={car_id}  steering={steering_joints}  motors={motor_joints}")

for body_id in range(p.getNumBodies()):
    if p.getDynamicsInfo(body_id, -1)[0] == 10.0:
        p.changeDynamics(body_id, -1, mass=0.0)
        print(f"  [Init] Body {body_id} → static")

for side_y in [ROAD_HALF_W, -ROAD_HALF_W]:
    wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[16.66, 0.02, 0.4])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wc,
                      baseVisualShapeIndex=-1,
                      basePosition=[16.66, side_y, 0.4])
print(f"[Init] Boundary walls at y=±{ROAD_HALF_W}\n")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
vout   = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (CAM_W, CAM_H))
if not vout.isOpened():
    raise RuntimeError("VideoWriter failed to open")
print(f"[Video] → {VIDEO_PATH}\n")

# ═══════════════════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════════════════
def get_frame():
    pos, orn = p.getBasePositionAndOrientation(car_id)
    R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    eye = np.array(pos) + np.array([0., 0., 0.3])
    vm = p.computeViewMatrix(eye.tolist(),
                              (eye + R @ np.array([1.,0.,0.])).tolist(),
                              (R @ np.array([0.,0.,1.])).tolist())
    pm = p.computeProjectionMatrixFOV(fov=CAM_FOV, aspect=CAM_W/CAM_H,
                                       nearVal=CAM_NEAR, farVal=CAM_FAR)
    _, _, rgba, _, _ = p.getCameraImage(CAM_W, CAM_H, vm, pm,
                                         renderer=p.ER_TINY_RENDERER)
    return cv2.cvtColor(
        np.array(rgba, dtype=np.uint8).reshape(CAM_H, CAM_W, 4),
        cv2.COLOR_RGBA2BGR)

# ═══════════════════════════════════════════════════════════════════════
# OPTICAL FLOW
# ═══════════════════════════════════════════════════════════════════════
def detect_features(gray):
    return cv2.goodFeaturesToTrack(gray, **FEAT_PARAMS)
# ---------- CUSTOM PYRAMIDAL LK ----------
WIN_SIZE = 7
MAX_ITER = 10
LEVELS   = 3

def build_pyramid(img, levels):
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid[::-1]

def lk_point(I1, I2, x, y):
    u, v = 0.0, 0.0
    h, w = I1.shape

    for _ in range(MAX_ITER):
        xi = int(round(x + u))
        yi = int(round(y + v))

        if (xi - WIN_SIZE < 0 or yi - WIN_SIZE < 0 or
            xi + WIN_SIZE + 1 > w or yi + WIN_SIZE + 1 > h):
            return None

        I1_win = I1[y-WIN_SIZE:y+WIN_SIZE+1, x-WIN_SIZE:x+WIN_SIZE+1].astype(np.float64)
        I2_win = I2[yi-WIN_SIZE:yi+WIN_SIZE+1, xi-WIN_SIZE:xi+WIN_SIZE+1].astype(np.float64)

        if I1_win.shape != I2_win.shape:
            return None

        Ix = cv2.Sobel(I1_win, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(I1_win, cv2.CV_64F, 0, 1, ksize=3)
        It = I2_win - I1_win

        A = np.vstack((Ix.flatten(), Iy.flatten())).T
        b = -It.flatten()
        ATA = A.T @ A

        if np.linalg.det(ATA) < 1e-4:
            return None

        nu = np.linalg.inv(ATA) @ (A.T @ b)
        du, dv = nu

        u += du
        v += dv

        if du*du + dv*dv < 0.01:
            break

    return x + u, y + v


def pyramidal_lk(I1, I2, points):
    pyr1 = build_pyramid(I1, LEVELS)
    pyr2 = build_pyramid(I2, LEVELS)

    results = []

    for pt in points:
        x, y = float(pt[0][0]), float(pt[0][1])
        u, v = 0.0, 0.0
        result = None

        for lvl in range(LEVELS):
            scale = 2 ** (LEVELS - lvl - 1)

            I1_lvl = pyr1[lvl]
            I2_lvl = pyr2[lvl]

            x_lvl = x / scale + u
            y_lvl = y / scale + v

            out = lk_point(I1_lvl, I2_lvl, int(round(x_lvl)), int(round(y_lvl)))
            if out is None:
                result = None
                break

            nx, ny = out
            u += nx - int(round(x_lvl))
            v += ny - int(round(y_lvl))

            if lvl != LEVELS - 1:
                u *= 2
                v *= 2

            result = (x + u, y + v)

        results.append(result)

    return results
def optical_flow(gp, gc, pp):
    if pp is None or len(pp) < 5:
        return None, None

    # Convert input points
    pts_in = np.array(pp, dtype=np.float32)

    new_pts = pyramidal_lk(gp, gc, pts_in)

    pts_prev = []
    pts_curr = []

    for old, new in zip(pts_in, new_pts):
        if new is None:
            continue

        x0, y0 = old[0]
        x1, y1 = new

        dx, dy = x1 - x0, y1 - y0
        dist2 = dx*dx + dy*dy

        # Same filtering logic as your standalone script
        if not (0.1 < dist2 < 5000):
            continue

        pts_prev.append([x0, y0])
        pts_curr.append([x1, y1])

    if len(pts_prev) < 5:
        return None, None

    pts_prev = np.array(pts_prev, dtype=np.float32).reshape(-1,1,2)
    pts_curr = np.array(pts_curr, dtype=np.float32).reshape(-1,1,2)

    return pts_prev, pts_curr
# ═══════════════════════════════════════════════════════════════════════
# FOCUS OF EXPANSION
# ═══════════════════════════════════════════════════════════════════════
def estimate_foe(pts_p, pts_c):
    cx, cy = CAM_W // 2, CAM_H // 2
    if pts_p is None or len(pts_p) < 6:
        return float(cx), float(cy)
    P  = pts_p.reshape(-1, 2).astype(float)
    C  = pts_c.reshape(-1, 2).astype(float)
    dF = C - P
    a = dF[:, 1]; b = -dF[:, 0]
    c = dF[:, 1]*P[:, 0] - dF[:, 0]*P[:, 1]
    best = np.array([cx, cy], dtype=float); best_n = 0
    rng = np.random.default_rng(7)
    for _ in range(100):
        i, j = rng.choice(len(a), 2, replace=False)
        det = a[i]*b[j] - a[j]*b[i]
        if abs(det) < 1e-7: continue
        x = (c[i]*b[j] - c[j]*b[i]) / det
        y = (a[i]*c[j] - a[j]*c[i]) / det
        if not (0 <= x <= CAM_W and 0 <= y <= CAM_H): continue
        n = int(np.sum((P[:,0]-x)*dF[:,0] + (P[:,1]-y)*dF[:,1] > 0))
        if n > best_n: best_n = n; best = np.array([x, y])
    return float(best[0]), float(best[1])

# ═══════════════════════════════════════════════════════════════════════
# OPTICAL FLOW LOOMING (Layer 1 — algorithm requirement)
# ═══════════════════════════════════════════════════════════════════════
def detect_looming(pts_p, pts_c, foe):
    if pts_p is None or len(pts_p) < 8:
        return 0.0, []
    fx, fy = foe
    P  = pts_p.reshape(-1, 2).astype(float)
    C  = pts_c.reshape(-1, 2).astype(float)
    dF = C - P; mag = np.linalg.norm(dF, axis=1)
    ego = P - np.array([fx, fy])
    ego_dir  = ego / (np.linalg.norm(ego, axis=1, keepdims=True) + 1e-6)
    flow_dir = dF  / (mag[:, None] + 1e-6)
    cos_sim  = np.sum(ego_dir * flow_dir, axis=1)
    mask = (cos_sim < 0.4) & (mag > 0.15)
    obs_P = P[mask]; obs_dF = dF[mask]
    if len(obs_P) < 4: return 0.0, []
    lower = obs_P[:, 1] > CAM_H // 4
    obs_P = obs_P[lower]; obs_dF = obs_dF[lower]
    if len(obs_P) < 3: return 0.0, []
    cx_c = obs_P[:, 0].mean(); cy_c = obs_P[:, 1].mean()
    out   = obs_P - np.array([cx_c, cy_c])
    out_n = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)
    loom  = float(np.mean(np.sum(obs_dF * out_n, axis=1)))
    if loom > 0.05:
        side = np.sign(cx_c - CAM_W//2)
        if side == 0: side = 1.0
        w = loom * (1.0 - abs(cx_c - CAM_W//2) / (CAM_W/2))
        return float(np.clip(-side * w * 1.5, -1.0, 1.0)), [(cx_c, cy_c, loom, side)]
    return 0.0, []

# ═══════════════════════════════════════════════════════════════════════
# BOUNDARY REPULSION
# ═══════════════════════════════════════════════════════════════════════
def boundary_repulsion(cy):
    force = 0.0
    dr = ROAD_HALF_W - cy
    dl = ROAD_HALF_W + cy
    if dr < BOUNDARY_SAFE:
        force -= ((BOUNDARY_SAFE - dr) / BOUNDARY_SAFE) ** 2
    if dl < BOUNDARY_SAFE:
        force += ((BOUNDARY_SAFE - dl) / BOUNDARY_SAFE) ** 2
    return float(np.clip(force, -1.0, 1.0))

# ═══════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════
CRUISE     = "CRUISE"
APPROACH   = "APPROACH"
AVOID      = "AVOID"
STRAIGHTEN = "STRAIGHTEN"

state       = CRUISE
active_obs  = None   # (ox, oy) of current obstacle being avoided

def update_state(cx, cy):
    global state, active_obs

    if state == CRUISE:
        # Look for the next obstacle ahead
        for (ox, oy) in OBS_POSITIONS:
            dx = ox - cx
            if 0 < dx <= APPROACH_X:
                state      = APPROACH
                active_obs = (ox, oy)
                print(f"  [State] CRUISE → APPROACH  obs=({ox},{oy:+.2f})  dx={dx:.1f}")
                break

    elif state == APPROACH:
        if active_obs:
            ox, oy = active_obs
            dx = ox - cx
            if dx <= 0:                          # reached obstacle X
                state = AVOID
                print(f"  [State] APPROACH → AVOID")
            elif dx > APPROACH_X + 0.5:          # somehow moved back
                state = CRUISE; active_obs = None

    elif state == AVOID:
        if active_obs:
            ox, oy = active_obs
            dx = ox - cx                         # negative when past obstacle
            if dx < -CLEARED_X:
                state = STRAIGHTEN
                print(f"  [State] AVOID → STRAIGHTEN  y={cy:+.3f}")

    elif state == STRAIGHTEN:
        if abs(cy) < STRAIGHT_TOL:              # close enough to centre
            state = CRUISE; active_obs = None
            print(f"  [State] STRAIGHTEN → CRUISE  y={cy:+.3f}")

    return state

def state_target_y():
    """Returns the Y target for the current state."""
    if state == CRUISE:
        return 0.0
    elif state in (APPROACH, AVOID):
        if active_obs:
            ox, oy = active_obs
            return CLEAR_Y[ox]
    return 0.0   # STRAIGHTEN → aim for centre

# ═══════════════════════════════════════════════════════════════════════
# CONTROL
# ═══════════════════════════════════════════════════════════════════════
def compute_force(cx, cy, loom_f, foe):
    target_y = state_target_y()
    b_f      = boundary_repulsion(cy)

    # P-controller toward target Y
    err      = target_y - cy
    y_force  = float(np.clip(err * 2.0, -1.0, 1.0))

    # FOE attraction: keeps heading straight, gentle correction
    foe_attract = (CAM_W//2 - foe[0]) / (CAM_W / 2.0)

    if state == CRUISE:
        force = 0.60*y_force + 0.20*foe_attract + 0.10*loom_f + 0.10*b_f
    elif state == STRAIGHTEN:
        # Strong pull back to centre, ignore loom
        force = 0.80*y_force + 0.10*foe_attract + 0.10*b_f
    else:
        # APPROACH / AVOID: steer hard to clear side
        force = 0.70*y_force + 0.15*loom_f + 0.10*b_f + 0.05*foe_attract

    return float(np.clip(force, -1.0, 1.0)), target_y, b_f

def compute_speed():
    if state in (APPROACH, AVOID):
        return AVOID_SPEED
    return BASE_SPEED

def apply_control(force, speed):
    steer = STEER_GAIN * force * MAX_STEER
    for j in steering_joints:
        p.setJointMotorControl2(car_id, j, controlMode=p.POSITION_CONTROL,
                                targetPosition=steer, force=10)
    for j in motor_joints:
        p.setJointMotorControl2(car_id, j, controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=speed, force=800)

def stop_car():
    for j in motor_joints:
        p.setJointMotorControl2(car_id, j, controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=0, force=800)
    for j in steering_joints:
        p.setJointMotorControl2(car_id, j, controlMode=p.POSITION_CONTROL,
                                targetPosition=0, force=10)

def get_pos():
    pos, _ = p.getBasePositionAndOrientation(car_id)
    return pos[0], pos[1]

def reached_goal():
    return get_pos()[0] >= GOAL_X

# ═══════════════════════════════════════════════════════════════════════
# OVERLAY
# ═══════════════════════════════════════════════════════════════════════
STATE_COL = {CRUISE:"(0,255,0)", APPROACH:"(0,200,255)",
             AVOID:"(0,0,255)",  STRAIGHTEN:"(255,200,0)"}
STATE_BGR = {CRUISE:(0,255,0), APPROACH:(0,200,255),
             AVOID:(0,0,255),   STRAIGHTEN:(255,200,0)}

def draw_overlay(frame, pts_p, pts_c, foe, loom_dbg,
                 force, loom_f, b_f, speed, target_y):
    vis = frame.copy()

    # Flow arrows
    if pts_p is not None and pts_c is not None:
        for (x0,y0),(x1,y1) in zip(pts_p.reshape(-1,2), pts_c.reshape(-1,2)):
            cv2.arrowedLine(vis,(int(x0),int(y0)),(int(x1),int(y1)),
                            (0,200,0),1,tipLength=0.5)

    # Looming circles
    for (ox,oy,loom,side) in loom_dbg:
        cv2.circle(vis,(int(ox),int(oy)),int(25+loom*50),(0,140,255),2)

    # FOE
    cv2.drawMarker(vis,(int(foe[0]),int(foe[1])),(255,0,255),cv2.MARKER_CROSS,28,2)

    # Centre line
    cv2.line(vis,(CAM_W//2,0),(CAM_W//2,CAM_H),(255,220,0),1)

    # Target Y line
    ty_px = int(CAM_H//2 * (1 - target_y / ROAD_HALF_W))
    ty_px = max(5, min(CAM_H-5, ty_px))
    cv2.line(vis,(0,ty_px),(CAM_W,ty_px),(0,255,255),2)

    # Steering bar
    mid  = CAM_W//2
    barx = int(mid + force*CAM_W//4)
    cv2.rectangle(vis,(mid,CAM_H-22),(barx,CAM_H-6),
                  (0,180,255) if force>=0 else (255,100,0),-1)
    cv2.line(vis,(mid,CAM_H-27),(mid,CAM_H-1),(255,255,255),1)

    # State badge
    col = STATE_BGR.get(state, (200,200,200))
    cv2.rectangle(vis,(CAM_W-160,0),(CAM_W,30),col,-1)
    cv2.putText(vis,state,(CAM_W-155,22),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)

    px,py = get_pos()
    # Edge warning
    if abs(py) > ROAD_HALF_W - BOUNDARY_SAFE:
        sx = CAM_W-8 if py>0 else 0
        cv2.rectangle(vis,(sx,0),(sx+8,CAM_H),(0,0,220),-1)

    for k, txt in enumerate([
        f"X:{px:.2f}/{GOAL_X}  Y:{py:+.3f}",
        f"Force:{force:+.3f}  Spd:{speed:.1f}",
        f"Tgt Y:{target_y:+.2f}  Loom:{loom_f:+.3f}  B:{b_f:+.3f}",
    ]):
        cv2.putText(vis,txt,(10,28+k*26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
    return vis

# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
print("="*55)
print("  Optical Flow Navigation — Lucas-Kanade")
print(f"  Goal: X>={GOAL_X}m  Road: y=±{ROAD_HALF_W}m")
print(f"  States: CRUISE → APPROACH → AVOID → STRAIGHTEN → CRUISE")
print(f"  Video → {VIDEO_PATH}")
print("="*55+"\n")

prev_gray=None; prev_pts=None; frame_num=0
foe=(float(CAM_W//2), float(CAM_H//2))
loom_dbg=[]; loom_f=0.0
force=0.0; speed=BASE_SPEED
last_x=0.0; last_x_t=time.time(); stuck_armed=False

try:
    while not reached_goal():
        p.stepSimulation()

        frame = get_frame()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pts_p, pts_c = None, None
        if prev_gray is not None and prev_pts is not None:
            pts_p, pts_c = optical_flow(prev_gray, gray, prev_pts)
        if pts_p is not None and len(pts_p) >= 6:
            foe = estimate_foe(pts_p, pts_c)
        loom_f, loom_dbg = detect_looming(pts_p, pts_c, foe)

        px, py = get_pos()
        update_state(px, py)
        force, target_y, b_f = compute_force(px, py, loom_f, foe)
        speed = compute_speed()
        apply_control(force, speed)

        if frame_num % 10 == 0 or prev_pts is None:
            prev_pts = detect_features(gray)
        else:
            prev_pts = (pts_c.reshape(-1,1,2)
                        if pts_c is not None else detect_features(gray))
        prev_gray = gray

        if frame_num % 2 == 0:
            vis = draw_overlay(frame,pts_p,pts_c,foe,loom_dbg,
                               force,loom_f,b_f,speed,target_y)
            vout.write(vis)
        frame_num += 1

        # Stuck detection
        cur_x = get_pos()[0]
        if cur_x >= 1.0: stuck_armed = True
        if stuck_armed:
            if abs(cur_x - last_x) > 0.08:
                last_x = cur_x; last_x_t = time.time()
            elif time.time() - last_x_t > 6.0:
                print(f"  [!] Stuck at X={cur_x:.2f} Y={py:.3f} — nudge to centre")
                apply_control(-np.sign(py)*0.5, BASE_SPEED)
                for _ in range(120): p.stepSimulation(); time.sleep(DT)
                last_x = get_pos()[0]; last_x_t = time.time()

        if frame_num % 60 == 0:
            print(f"  t={frame_num//60:3d}s | X={px:5.2f} Y={py:+.4f} "
                  f"| {state:12s} | F={force:+.3f} L={loom_f:+.3f} "
                  f"B={b_f:+.3f} spd={speed:.1f}")

        time.sleep(DT)

    print("\n✓  Goal reached!")

except KeyboardInterrupt:
    print(f"\n  Aborted after {frame_num} frames.")

finally:
    stop_car(); vout.release(); p.disconnect(); _display.stop()
    size = os.path.getsize(VIDEO_PATH) if os.path.exists(VIDEO_PATH) else 0
    print(f"\n[Done] {VIDEO_PATH}  ({size//1024} KB)")
    print("[Watch] cd /tmp && python3 -m http.server 8080") 
