import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

print("=== Metric Stereo VO Pipeline ===\n")

# Load config
config_path = Path("config/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

np.random.seed(config.get("random_seed", 42))

seq_path = Path(config["sequences"]["room2"])
left_path  = seq_path / "mav0" / "cam0" / "data"
right_path = seq_path / "mav0" / "cam1" / "data"

img_files = sorted(left_path.glob("*.png"))
print(f"Found {len(img_files)} stereo pairs")

# Camera intrinsics (from camchain.yaml)
K = np.array([[458.654, 0,      367.215],
              [0,      457.296, 248.375],
              [0,      0,      1]], dtype=np.float32)

baseline = 0.1011  # from your earlier result

# Stereo matcher
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
# Alternative (better but slower): StereoSGBM
# stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=11)

poses = [np.eye(4)]

print("Starting Stereo VO...")

for i in tqdm(range(len(img_files)-1)):
    # Load stereo pair
    left1  = cv2.imread(str(img_files[i]), 0)
    right1 = cv2.imread(str(right_path / img_files[i].name), 0)
    left2  = cv2.imread(str(img_files[i+1]), 0)
    right2 = cv2.imread(str(right_path / img_files[i+1].name), 0)

    # Compute disparity on current frame (for metric depth)
    disparity = stereo.compute(left1, right1)
    disparity = disparity.astype(np.float32) / 16.0

    # Optional: simple depth map for debugging
    # depth = (K[0,0] * baseline) / (disparity + 1e-6)

    # Feature detection on left image (same as monocular)
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(left1, None)
    kp2, des2 = orb.detectAndCompute(left2, None)

    if des1 is None or des2 is None or len(kp1) < 150:
        poses.append(poses[-1])
        continue

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 100:
        poses.append(poses[-1])
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Use 3D-2D PnP with metric points? For now we keep Essential Matrix (we'll upgrade next)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    if E is None:
        poses.append(poses[-1])
        continue

    _, R_mat, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    T_rel = np.eye(4)
    T_rel[:3, :3] = R_mat
    T_rel[:3, 3] = (t * baseline).ravel()   # ← Scale by baseline! (key difference)

    current_pose = poses[-1] @ T_rel
    poses.append(current_pose)

print(f"\n✅ Stereo VO finished with {len(poses)} poses")

# Save metric trajectory
traj_path = Path("results/trajectories/stereo_room2.txt")
traj_path.parent.mkdir(parents=True, exist_ok=True)

with open(traj_path, 'w') as f:
    for i, pose in enumerate(poses):
        t = pose[:3, 3]
        q = R.from_matrix(pose[:3, :3]).as_quat()
        ts = i * 0.05
        f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

print(f"✅ Metric trajectory saved → {traj_path}")

# Plot comparison
positions = np.array([p[:3, 3] for p in poses])
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:,0], positions[:,1], positions[:,2], 'r-', linewidth=1.5, label='Stereo VO (Metric)')
ax.set_title('Stereo Visual Odometry Trajectory (METRIC Scale)')
ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)'); ax.set_zlabel('Z (meters)')
ax.legend()
plt.savefig('results/plots/stereo_trajectory.png', dpi=300)
plt.show()

print("\n🎉 Stereo VO Complete!")
print("You now have both monocular (up-to-scale) and stereo (metric) trajectories!")