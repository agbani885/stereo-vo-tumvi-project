import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

print("=== Full Monocular VO Pipeline ===\n")

# Load config
config_path = Path("config/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

np.random.seed(config.get("random_seed", 42))

seq_path = Path(config["sequences"]["room2"])
left_path = seq_path / "mav0" / "cam0" / "data"
img_files = sorted(left_path.glob("*.png"))

# Camera intrinsics from previous step
K = np.array([[458.654, 0, 367.215],
              [0, 457.296, 248.375],
              [0, 0, 1]], dtype=np.float32)   # Approximate from TUM VI cam0

orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Initialize trajectory
poses = [np.eye(4)]   # First pose = identity
prev_kp, prev_des = None, None

print(f"Processing {len(img_files)} frames...")

for i in tqdm(range(len(img_files)-1)):
    img1 = cv2.imread(str(img_files[i]), 0)
    img2 = cv2.imread(str(img_files[i+1]), 0)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 100:
        poses.append(poses[-1])
        continue

    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 80:
        poses.append(poses[-1])
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is None or E.shape != (3,3):
        poses.append(poses[-1])
        continue

    _, R_mat, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Build transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t.ravel()

    # Accumulate pose
    current_pose = poses[-1] @ T
    poses.append(current_pose)

print(f"\n✅ Computed {len(poses)} poses")

# Save trajectory in TUM format
traj_path = Path("results/trajectories/monocular_room2.txt")
traj_path.parent.mkdir(parents=True, exist_ok=True)

with open(traj_path, 'w') as f:
    for i, pose in enumerate(poses):
        t = pose[:3, 3]
        q = R.from_matrix(pose[:3, :3]).as_quat()  # x,y,z,w
        timestamp = i * 0.05  # approx 20Hz
        f.write(f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

print(f"✅ Trajectory saved to {traj_path}")

# Simple 3D plot
positions = np.array([p[:3, 3] for p in poses])
plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot(positions[:,0], positions[:,1], positions[:,2], 'b-')
ax.set_title('Monocular VO Trajectory (Up-to-Scale) - Room2')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.savefig('results/plots/monocular_trajectory.png')
plt.show()

print("\n🎉 Monocular VO Frontend Complete!")
print("Next: Add triangulation + Bundle Adjustment + Stereo Metric Scale")
