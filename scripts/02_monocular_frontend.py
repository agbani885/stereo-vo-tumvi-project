import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("=== Monocular VO Frontend: ORB + Essential Matrix ===\n")

# ====================== CONFIG ======================
config_path = Path("config/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

np.random.seed(config.get("random_seed", 42))

seq_path = Path(config["sequences"]["room2"])
left_path = seq_path / "mav0" / "cam0" / "data"

# Load image list
img_files = sorted(left_path.glob("*.png"))
print(f"Found {len(img_files)} images in left camera")

# ====================== ORB DETECTOR ======================
orb = cv2.ORB_create(
    nfeatures=2000,          # number of features
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20
)

# BFMatcher with ratio test
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

poses = []          # list of (R, t) up-to-scale
prev_kp = None
prev_des = None

print("\nStarting frame-to-frame tracking...")

for i in tqdm(range(len(img_files)-1)):
    # Load consecutive images
    img1 = cv2.imread(str(img_files[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img_files[i+1]), cv2.IMREAD_GRAYSCALE)
    
    # Detect features
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(kp1) < 50 or len(kp2) < 50:
        print(f"⚠️  Not enough features on frame {i}")
        continue
    
    # Match features
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Ratio test (Lowe's ratio)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 50:
        print(f"⚠️  Too few good matches on frame {i}: {len(good_matches)}")
        continue
    
    # Get corresponding points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Essential Matrix + RANSAC
    E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, 
                                   prob=0.999, threshold=1.0)
    
    if E is None:
        continue
    
    # Recover pose (4 possible solutions)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2)
    
    poses.append((R, t))
    
    # Optional: visualize matches (first 10 frames only)
    if i < 10:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(f"Matches Frame {i}", match_img)
        cv2.waitKey(100)  # 100ms delay

print(f"\n✅ Processed {len(poses)} frames")
print(f"Final pose count: {len(poses)}")

# Save first 5 poses for inspection
print("\nFirst 3 translations (up-to-scale):")
for j in range(min(3, len(poses))):
    R, t = poses[j]
    print(f"Frame {j} -> {j+1}: t = [{t[0][0]:.4f}, {t[1][0]:.4f}, {t[2][0]:.4f}]")

cv2.destroyAllWindows()
print("\n🎉 Monocular Frontend Complete!")
print("Next: Triangulation + Full Trajectory + Stereo Extension")