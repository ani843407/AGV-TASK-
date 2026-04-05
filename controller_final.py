"""
controller.py
=============
Optical-Flow Navigation  —  Lucas-Kanade method

Fixes:
  1. clear_y sign was inverted — obs_col on LEFT means obstacle is LEFT,
     clear side is RIGHT (+0.55), not LEFT (-0.55)
  2. APPROACH → AVOID transition now uses a time/distance window instead
     of relying on exact dx<=0 which was never clean
  3. APPROACH entered immediately on first confirmed detection,
     obstacle X estimate updated continuously while approaching
  4. Once in AVOID, car drives forward regardless of detection

Run:
    xvfb-run -a python3 controller.py
"""

from pyvirtualdisplay import Display
_display = Display(visible=0, size=(1280, 720))
_display.start()

import os
import cv2
import numpy as np
import pybullet as p
from simulation_setup import setup_simulation

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
CAM_W, CAM_H  = 640, 480
CAM_FOV       = 70
CAM_NEAR      = 0.1
CAM_FAR       = 25.0
FPS           = 30
DT            = 1.0 / 60.0
SIM_STEPS_PER_FRAME = 2

GOAL_X        = 28.0
BASE_SPEED    = 10.0
AVOID_SPEED   = 6.0
MAX_STEER     = 0.5
STEER_GAIN    = 0.65

ROAD_HALF_W   = 1.16
BOUNDARY_SAFE = 0.45

OBS_POSITIONS = []
CLEAR_Y       = {}

LOOM_THRESHOLD      = 0.25
LOOM_CONFIRM_FRAMES = 4
MIN_DIST_BETWEEN_OBS= 3.0

# APPROACH → AVOID after car has closed this much distance since detection
AVOID_AFTER_TRAVEL  = 2.0   # metres — once we've moved 2m since detection, switch to AVOID
# AVOID → STRAIGHTEN after car has travelled this far past the estimated obs X
CLEARED_X           = 2.0
STRAIGHT_TOL        = 0.12

LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FEAT_PARAMS = dict(maxCorners=200, qualityLevel=0.005, minDistance=6, blockSize=7)
VIDEO_PATH   = "/tmp/sim_video.avi"

# ═══════════════════════════════════════════════════════════════════════
# SIMULATION INIT
# ═══════════════════════════════════════════════════════════════════════
car_id, steering_joints, motor_joints = setup_simulation(
    dt=DT, settle_frames=60, gui=True)
print(f"\n[Init] car_id={car_id}  steering={steering_joints}  motors={motor_joints}")

for body_id in range(p.getNumBodies()):
    if p.getDynamicsInfo(body_id, -1)[0] == 10.0:
        p.changeDynamics(body_id, -1, mass=0.0)
        print(f"  [Init] Body {body_id} → static obstacle")

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
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def get_pos():
    pos, _ = p.getBasePositionAndOrientation(car_id)
    return pos[0], pos[1]

def get_vx():
    lin, _ = p.getBaseVelocity(car_id)
    return lin[0]

def reached_goal():
    return get_pos()[0] >= GOAL_X

# ═══════════════════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════════════════
def get_frame():
    pos, orn = p.getBasePositionAndOrientation(car_id)
    R   = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    eye = np.array(pos) + np.array([0., 0., 0.3])
    vm  = p.computeViewMatrix(eye.tolist(),
                               (eye + R@np.array([1.,0.,0.])).tolist(),
                               (R@np.array([0.,0.,1.])).tolist())
    pm  = p.computeProjectionMatrixFOV(fov=CAM_FOV, aspect=CAM_W/CAM_H,
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
    mask = np.zeros_like(gray)
    mask[int(CAM_H*0.35):int(CAM_H*0.90),
         int(CAM_W*0.15):int(CAM_W*0.85)] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, **FEAT_PARAMS)

WIN_SIZE=7; MAX_ITER=10; LEVELS=3

def build_pyramid(img, levels):
    pyr=[img]
    for _ in range(levels-1): img=cv2.pyrDown(img); pyr.append(img)
    return pyr[::-1]

def lk_point(I1, I2, x, y):
    u,v=0.0,0.0; h,w=I1.shape
    for _ in range(MAX_ITER):
        xi=int(round(x+u)); yi=int(round(y+v))
        if xi-WIN_SIZE<0 or yi-WIN_SIZE<0 or xi+WIN_SIZE+1>w or yi+WIN_SIZE+1>h:
            return None
        I1w=I1[y-WIN_SIZE:y+WIN_SIZE+1,x-WIN_SIZE:x+WIN_SIZE+1].astype(np.float64)
        I2w=I2[yi-WIN_SIZE:yi+WIN_SIZE+1,xi-WIN_SIZE:xi+WIN_SIZE+1].astype(np.float64)
        if I1w.shape!=I2w.shape: return None
        Ix=cv2.Sobel(I1w,cv2.CV_64F,1,0,ksize=3)
        Iy=cv2.Sobel(I1w,cv2.CV_64F,0,1,ksize=3)
        It=I2w-I1w
        A=np.vstack((Ix.flatten(),Iy.flatten())).T; bv=-It.flatten()
        ATA=A.T@A
        if np.linalg.det(ATA)<1e-4: return None
        nu=np.linalg.inv(ATA)@(A.T@bv); u+=nu[0]; v+=nu[1]
        if nu[0]**2+nu[1]**2<0.01: break
    return x+u,y+v

def pyramidal_lk(I1, I2, points):
    pyr1=build_pyramid(I1,LEVELS); pyr2=build_pyramid(I2,LEVELS)
    results=[]
    for pt in points:
        x,y=float(pt[0][0]),float(pt[0][1]); u,v=0.0,0.0; result=None
        for lvl in range(LEVELS):
            scale=2**(LEVELS-lvl-1); xl=x/scale+u; yl=y/scale+v
            out=lk_point(pyr1[lvl],pyr2[lvl],int(round(xl)),int(round(yl)))
            if out is None: result=None; break
            nx,ny=out; u+=nx-int(round(xl)); v+=ny-int(round(yl))
            if lvl!=LEVELS-1: u*=2; v*=2
            result=(x+u,y+v)
        results.append(result)
    return results

def optical_flow(gp, gc, pp):
    if pp is None or len(pp)<5: return None,None
    pts_in=np.array(pp,dtype=np.float32)
    new_pts=pyramidal_lk(gp,gc,pts_in)
    prev_out,curr_out=[],[]
    for old,new in zip(pts_in,new_pts):
        if new is None: continue
        x0,y0=old[0]; x1,y1=new
        d2=(x1-x0)**2+(y1-y0)**2
        if not(0.1<d2<5000): continue
        prev_out.append([x0,y0]); curr_out.append([x1,y1])
    if len(prev_out)<5: return None,None
    return (np.array(prev_out,dtype=np.float32).reshape(-1,1,2),
            np.array(curr_out,dtype=np.float32).reshape(-1,1,2))

# ═══════════════════════════════════════════════════════════════════════
# FOCUS OF EXPANSION
# ═══════════════════════════════════════════════════════════════════════
def estimate_foe(pts_p, pts_c):
    cx,cy=CAM_W//2,CAM_H//2
    if pts_p is None or len(pts_p)<6: return float(cx),float(cy)
    P=pts_p.reshape(-1,2).astype(float); C=pts_c.reshape(-1,2).astype(float); dF=C-P
    a=dF[:,1]; b=-dF[:,0]; c=dF[:,1]*P[:,0]-dF[:,0]*P[:,1]
    best=np.array([cx,cy],dtype=float); best_n=0; rng=np.random.default_rng(7)
    for _ in range(100):
        i,j=rng.choice(len(a),2,replace=False); det=a[i]*b[j]-a[j]*b[i]
        if abs(det)<1e-7: continue
        x=(c[i]*b[j]-c[j]*b[i])/det; y=(a[i]*c[j]-a[j]*c[i])/det
        if not(0<=x<=CAM_W and 0<=y<=CAM_H): continue
        n=int(np.sum((P[:,0]-x)*dF[:,0]+(P[:,1]-y)*dF[:,1]>0))
        if n>best_n: best_n=n; best=np.array([x,y])
    return float(best[0]),float(best[1])

# ═══════════════════════════════════════════════════════════════════════
# LOOMING DETECTION
# ═══════════════════════════════════════════════════════════════════════
def detect_looming(pts_p, pts_c, foe):
    """
    Returns (loom_score, obs_col, debug_list)
    obs_col: pixel column of obstacle cluster
      < CAM_W/2 → obstacle is on LEFT  → clear side is RIGHT → target_y = +0.55
      > CAM_W/2 → obstacle is on RIGHT → clear side is LEFT  → target_y = -0.55
    """
    if pts_p is None or len(pts_p) < 6:
        return 0.0, None, []

    fx, fy = foe
    P  = pts_p.reshape(-1, 2).astype(float)
    C  = pts_c.reshape(-1, 2).astype(float)
    dF = C - P
    mag = np.linalg.norm(dF, axis=1)

    ego     = P - np.array([fx, fy])
    ego_dir = ego / (np.linalg.norm(ego, axis=1, keepdims=True) + 1e-6)
    flow_dir= dF / (mag[:, None] + 1e-6)
    cos_sim = np.sum(ego_dir * flow_dir, axis=1)

    in_zone = ((P[:, 1] > CAM_H * 0.40) &
               (P[:, 1] < CAM_H * 0.85) &
               (P[:, 0] > CAM_W * 0.15) &
               (P[:, 0] < CAM_W * 0.85))

    obs_mask = (cos_sim < 0.45) & (mag > 0.2) & in_zone
    obs_P    = P[obs_mask]
    obs_dF   = dF[obs_mask]

    if len(obs_P) < 4:
        return 0.0, None, []

    cx_c = obs_P[:, 0].mean()
    cy_c = obs_P[:, 1].mean()
    out   = obs_P - np.array([cx_c, cy_c])
    out_n = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-6)
    loom  = float(np.mean(np.sum(obs_dF * out_n, axis=1)))

    if loom > LOOM_THRESHOLD:
        return loom, cx_c, [(cx_c, cy_c, loom)]

    return loom, None, []

# ═══════════════════════════════════════════════════════════════════════
# OBSTACLE COLUMN → CLEAR SIDE Y
# ═══════════════════════════════════════════════════════════════════════
def col_to_clear_y(obs_col):
    """
    obs_col < CAM_W/2 → obstacle is on LEFT  → steer RIGHT → +0.55
    obs_col > CAM_W/2 → obstacle is on RIGHT → steer LEFT  → -0.55
    """
    if obs_col < CAM_W / 2:
        return +0.55   # obstacle left, go right
    else:
        return -0.55   # obstacle right, go left

# ═══════════════════════════════════════════════════════════════════════
# BOUNDARY REPULSION
# ═══════════════════════════════════════════════════════════════════════
def boundary_repulsion(cy):
    force=0.0
    dr=ROAD_HALF_W-cy; dl=ROAD_HALF_W+cy
    if dr<BOUNDARY_SAFE: s=(BOUNDARY_SAFE-dr)/BOUNDARY_SAFE; force-=s*s*s
    if dl<BOUNDARY_SAFE: s=(BOUNDARY_SAFE-dl)/BOUNDARY_SAFE; force+=s*s*s
    return float(np.clip(force,-1.0,1.0))

# ═══════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════
CRUISE="CRUISE"; APPROACH="APPROACH"; AVOID="AVOID"; STRAIGHTEN="STRAIGHTEN"
state       = CRUISE
active_obs  = None    # (est_x, clear_y)
detect_x    = None    # car X when obstacle was first detected

def try_register(car_x, obs_col, loom_score):
    """Register a new obstacle if far enough from existing ones."""
    global OBS_POSITIONS, CLEAR_Y
    for ox, _ in OBS_POSITIONS:
        if abs(ox - car_x) < MIN_DIST_BETWEEN_OBS:
            return None
    est_x   = car_x + 2.5+3
    clear_y = col_to_clear_y(obs_col)
    OBS_POSITIONS.append((est_x, clear_y))
    CLEAR_Y[est_x] = clear_y
    print(f"  [Detect] Obs registered x≈{est_x:.1f} "
          f"col={obs_col:.0f} ({'L' if obs_col < CAM_W/2 else 'R'}) "
          f"→ clear_y={clear_y:+.2f} loom={loom_score:.3f}")
    return (est_x, clear_y)

def update_state(car_x, car_y, obs_col, loom_score, confirmed):
    global state, active_obs, detect_x

    if state == CRUISE:
        if confirmed and obs_col is not None:
            obs = try_register(car_x, obs_col, loom_score)
            if obs is not None:
                active_obs = obs
                detect_x   = car_x
                state      = APPROACH
                print(f"  [State] CRUISE→APPROACH  tgt_y={obs[1]:+.2f}  detect_x={car_x:.2f}")

    elif state == APPROACH:
        if active_obs:
            # Transition to AVOID once car has travelled AVOID_AFTER_TRAVEL metres
            # since detection — don't rely on exact obstacle X estimate
            if detect_x is not None and (car_x - detect_x) >= AVOID_AFTER_TRAVEL:
                state = AVOID
                print(f"  [State] APPROACH→AVOID  y={car_y:+.3f}  travelled={car_x-detect_x:.2f}m")

    elif state == AVOID:
        if active_obs:
            ox, _ = active_obs
            # Clear once car is CLEARED_X past estimated obstacle position
            if car_x > ox + CLEARED_X:
                state      = STRAIGHTEN
                active_obs = None
                detect_x   = None
                print(f"  [State] AVOID→STRAIGHTEN  y={car_y:+.3f}")

    elif state == STRAIGHTEN:
        if abs(car_y) < STRAIGHT_TOL:
            state    = CRUISE
            active_obs = None
            detect_x   = None
            print(f"  [State] STRAIGHTEN→CRUISE  y={car_y:+.3f}")

def state_target_y():
    if state in (APPROACH, AVOID) and active_obs:
        return active_obs[1]
    return 0.0

# ═══════════════════════════════════════════════════════════════════════
# FORCE + SPEED
# ═══════════════════════════════════════════════════════════════════════
def compute_force(cy, loom_f, foe):
    target_y = state_target_y()
    b_f      = boundary_repulsion(cy)
    y_force  = float(np.clip((target_y - cy) * 2.0, -1.0, 1.0))
    foe_att  = (CAM_W//2 - foe[0]) / (CAM_W / 2.0)

    if state == STRAIGHTEN:
        force = y_force
        b_f   = 0.0
    elif state == CRUISE:
        force = 0.55*y_force + 0.20*foe_att + 0.10*loom_f + 0.15*b_f
    else:  # APPROACH / AVOID
        force = 0.65*y_force + 0.15*loom_f + 0.10*b_f + 0.10*foe_att

    return float(np.clip(force,-1.0,1.0)), target_y, b_f if state!=STRAIGHTEN else 0.0

def compute_speed():
    if state in (APPROACH, AVOID): return AVOID_SPEED
    return BASE_SPEED

def apply_control(force, speed):
    steer=STEER_GAIN*force*MAX_STEER
    for j in steering_joints:
        p.setJointMotorControl2(car_id,j,controlMode=p.POSITION_CONTROL,
                                targetPosition=steer,force=10)
    for j in motor_joints:
        p.setJointMotorControl2(car_id,j,controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=speed,force=800)

def stop_car():
    for j in motor_joints:
        p.setJointMotorControl2(car_id,j,controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=0,force=800)
    for j in steering_joints:
        p.setJointMotorControl2(car_id,j,controlMode=p.POSITION_CONTROL,
                                targetPosition=0,force=10)

# ═══════════════════════════════════════════════════════════════════════
# OVERLAY
# ═══════════════════════════════════════════════════════════════════════
STATE_BGR={CRUISE:(0,220,0),APPROACH:(0,200,255),AVOID:(0,0,255),STRAIGHTEN:(0,180,255)}

def draw_overlay(frame,pts_p,pts_c,foe,loom_dbg,force,loom_f,b_f,
                 speed,target_y,loom_conf,obs_col):
    vis=frame.copy()
    cv2.rectangle(vis,(int(CAM_W*0.15),int(CAM_H*0.40)),
                  (int(CAM_W*0.85),int(CAM_H*0.85)),(100,100,255),1)
    if pts_p is not None and pts_c is not None:
        for (x0,y0),(x1,y1) in zip(pts_p.reshape(-1,2),pts_c.reshape(-1,2)):
            cv2.arrowedLine(vis,(int(x0),int(y0)),(int(x1),int(y1)),
                            (0,200,0),1,tipLength=0.5)
    for (ox,oy,loom) in loom_dbg:
        cv2.circle(vis,(int(ox),int(oy)),int(20+loom*40),(0,140,255),2)
        cv2.putText(vis,f"{loom:.2f}",(int(ox)+5,int(oy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,140,255),1)
    if obs_col is not None:
        col_color=(0,0,255) if loom_conf>=LOOM_CONFIRM_FRAMES else (0,100,255)
        cv2.line(vis,(int(obs_col),int(CAM_H*0.40)),(int(obs_col),int(CAM_H*0.85)),
                 col_color,2)
        side_str = "LEFT" if obs_col < CAM_W/2 else "RIGHT"
        clear_str= f"→ go {'RIGHT' if obs_col < CAM_W/2 else 'LEFT'}"
        cv2.putText(vis,f"OBS {side_str} {clear_str}",
                    (int(obs_col)-60, int(CAM_H*0.40)-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,col_color,1)
    cv2.drawMarker(vis,(int(foe[0]),int(foe[1])),(255,0,255),cv2.MARKER_CROSS,28,2)
    cv2.line(vis,(CAM_W//2,0),(CAM_W//2,CAM_H),(255,220,0),1)
    ty_px=max(5,min(CAM_H-5,int(CAM_H//2-target_y*(CAM_H//2)/ROAD_HALF_W)))
    cv2.line(vis,(0,ty_px),(CAM_W,ty_px),(0,255,255),2)
    mid=CAM_W//2; barx=int(mid+force*CAM_W//4)
    cv2.rectangle(vis,(mid,CAM_H-22),(barx,CAM_H-6),
                  (0,180,255) if force>=0 else (255,100,0),-1)
    cv2.line(vis,(mid,CAM_H-27),(mid,CAM_H-1),(255,255,255),1)
    col=STATE_BGR.get(state,(180,180,180))
    cv2.rectangle(vis,(CAM_W-175,0),(CAM_W,32),col,-1)
    cv2.putText(vis,state,(CAM_W-170,23),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)
    # Confidence bar
    conf_w=int(min(loom_conf/LOOM_CONFIRM_FRAMES,1.0)*80)
    cv2.rectangle(vis,(CAM_W-175,34),(CAM_W-175+conf_w,44),(0,200,255),-1)
    px,py=get_pos()
    if abs(py)>ROAD_HALF_W-BOUNDARY_SAFE:
        sx=CAM_W-8 if py>0 else 0
        cv2.rectangle(vis,(sx,0),(sx+8,CAM_H),(0,0,220),-1)
    trav = f"trav:{px-detect_x:.1f}m" if detect_x is not None else ""
    for k,txt in enumerate([
        f"X:{px:.2f}/{GOAL_X}  Y:{py:+.3f}  tgt:{target_y:+.2f}",
        f"Force:{force:+.3f}  Spd:{speed:.1f}  B:{b_f:+.3f}",
        f"Loom:{loom_f:.3f}  conf:{loom_conf}  {trav}",
        f"Detected obs:{len(OBS_POSITIONS)}",
    ]):
        cv2.putText(vis,txt,(10,28+k*22),cv2.FONT_HERSHEY_SIMPLEX,0.47,(255,255,255),2)
    return vis

# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
print("="*55)
print("  Optical Flow Navigation — vision-based")
print(f"  APPROACH→AVOID after travelling {AVOID_AFTER_TRAVEL}m since detection")
print(f"  clear_y: col<{CAM_W//2}→+0.55(right)  col>{CAM_W//2}→-0.55(left)")
print(f"  Video → {VIDEO_PATH}")
print("="*55+"\n")

prev_gray=None; prev_pts=None; frame_num=0
foe=(float(CAM_W//2),float(CAM_H//2))
loom_dbg=[]; loom_f=0.0; loom_obs_col=None
force=0.0; speed=BASE_SPEED; b_f=0.0; target_y=0.0
loom_confidence=0  # OUTSIDE loop — persists across frames

low_vx_frames=0; stuck_armed=False; cooldown=0
STUCK_LIMIT=FPS*3; STUCK_COOL=FPS*4
recovering=False; rec_phase=0; rec_frames=0
REVERSE_F=int(FPS*0.5); STEER_F=int(FPS*1.2)

try:
    while not reached_goal():
        for _ in range(SIM_STEPS_PER_FRAME):
            p.stepSimulation()

        frame=get_frame()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        pts_p,pts_c=None,None
        if prev_gray is not None and prev_pts is not None:
            pts_p,pts_c=optical_flow(prev_gray,gray,prev_pts)
        if pts_p is not None and len(pts_p)>=6:
            foe=estimate_foe(pts_p,pts_c)
        loom_f,loom_obs_col,loom_dbg=detect_looming(pts_p,pts_c,foe)

        # Confidence counter — outside loop so it persists
        if loom_obs_col is not None:
            loom_confidence=min(loom_confidence+1, LOOM_CONFIRM_FRAMES+2)
        else:
            loom_confidence=max(loom_confidence-1, 0)

        confirmed = loom_confidence >= LOOM_CONFIRM_FRAMES

        px,py=get_pos(); vx=get_vx()

        if not recovering:
            update_state(px, py, loom_obs_col, loom_f, confirmed)
            force,target_y,b_f=compute_force(py,loom_f,foe)
            speed=compute_speed()
            apply_control(force,speed)
        else:
            rec_frames+=1
            if rec_phase==0:
                apply_control(0.0,-BASE_SPEED*0.5)
                if rec_frames>=REVERSE_F:
                    rec_phase=1; rec_frames=0
                    print("  [Recovery] Reversing → steering")
            else:
                apply_control(float(np.clip(-py*2.0,-1.0,1.0)),BASE_SPEED)
                if rec_frames>=STEER_F:
                    recovering=False; rec_phase=0; rec_frames=0
                    cooldown=STUCK_COOL; low_vx_frames=0
                    print("  [Recovery] Done")

        if frame_num%10==0 or prev_pts is None:
            prev_pts=detect_features(gray)
        else:
            prev_pts=(pts_c.reshape(-1,1,2)
                      if pts_c is not None else detect_features(gray))
        prev_gray=gray

        vis=draw_overlay(frame,pts_p,pts_c,foe,loom_dbg,force,loom_f,b_f,
                         speed,target_y,loom_confidence,loom_obs_col)
        vout.write(vis)
        frame_num+=1

        if px>2.0: stuck_armed=True
        if stuck_armed and not recovering:
            if cooldown>0: cooldown-=1
            else:
                if vx<0.3: low_vx_frames+=1
                else:       low_vx_frames=0
                if low_vx_frames>=STUCK_LIMIT:
                    print(f"  [!] Stuck X={px:.2f} Y={py:.3f} vx={vx:.2f}")
                    recovering=True; rec_phase=0; rec_frames=0; low_vx_frames=0

        if frame_num%FPS==0:
            trav=f" trav={px-detect_x:.2f}m" if detect_x else ""
            print(f"  t={frame_num//FPS:3d}s | X={px:5.2f} Y={py:+.4f} vx={vx:.2f} "
                  f"| {state:12s}{trav} | F={force:+.3f} tgt={target_y:+.2f} "
                  f"| loom={loom_f:.3f} conf={loom_confidence} "
                  f"obs={len(OBS_POSITIONS)} spd={speed:.1f}")

    print("\n✓  Goal reached!")

except KeyboardInterrupt:
    print(f"\n  Aborted after {frame_num} frames.")

finally:
    stop_car(); vout.release(); p.disconnect(); _display.stop()
    print(f"\nObstacles detected by vision: {len(OBS_POSITIONS)}")
    for i,(ox,oy) in enumerate(OBS_POSITIONS):
        print(f"  {i+1}. est_x≈{ox:.1f}  clear_y={oy:+.2f}")
    size=os.path.getsize(VIDEO_PATH) if os.path.exists(VIDEO_PATH) else 0
    print(f"\n[Done] {VIDEO_PATH}  ({size//1024} KB, {frame_num} frames @ {FPS}fps)")
    print("[Watch] cd /tmp && python3 -m http.server 8080")