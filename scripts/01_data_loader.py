import yaml
import cv2
import numpy as np
from pathlib import Path

print("=== Stereo VO Data Loader + Calibration ===\n")

# ====================== CONFIG ======================
config_path = Path("config/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

np.random.seed(config.get("random_seed", 42))

print("✅ Config loaded successfully!")
print(f"Random seed: {config.get('random_seed')}")
print(f"Feature type: {config.get('feature_type')}")

# ====================== PATHS ======================
seq_path = Path(config["sequences"]["room2"])
left_path  = seq_path / "mav0" / "cam0" / "data"
right_path = seq_path / "mav0" / "cam1" / "data"

print(f"\n✅ Looking for images in:")
print(f"Left  (cam0): {left_path}")
print(f"Right (cam1): {right_path}")

# Load first image pair
img_files = sorted(left_path.glob("*.png"))
if img_files:
    img_path = img_files[0]
    first_left = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    first_right = cv2.imread(str(right_path / img_path.name), cv2.IMREAD_GRAYSCALE)
    
    print(f"\n✅ First image pair loaded!")
    print(f"Image shape: {first_left.shape} (512x512 expected)")
else:
    print("❌ No images found!")
    exit()

# ====================== LOAD KALIBR CALIBRATION ======================
camchain_path = seq_path / "dso" / "camchain.yaml"

if camchain_path.exists():
    with open(camchain_path, 'r') as f:
        camchain = yaml.safe_load(f)

    # Left camera intrinsics
    cam0 = camchain.get('cam0', {})
    intrinsics = cam0.get('intrinsics', [0,0,0,0])
    fx = intrinsics[0]
    fy = intrinsics[1]
    cx = intrinsics[2]
    cy = intrinsics[3]

    print("\n✅ Kalibr Calibration Loaded!")
    print(f"Focal length fx: {fx:.2f} pixels")
    print(f"Focal length fy: {fy:.2f} pixels")
    print(f"Principal point cx: {cx:.2f}")
    print(f"Principal point cy: {cy:.2f}")

    # Stereo Baseline B (most important for metric Stereo VO)
    if 'cam1' in camchain and 'T_cn_cnm1' in camchain['cam1']:
        T = camchain['cam1']['T_cn_cnm1']
        baseline = abs(T[0][3])          # x-translation = baseline
        print(f"\n✅ Stereo Baseline B = {baseline:.4f} meters")
        print("   → This gives us METRIC scale in Stereo VO (PDF Section V)")
    else:
        print("⚠️  Stereo extrinsics not found")
else:
    print(f"❌ camchain.yaml not found at:\n   {camchain_path}")

print("\n🎉 Data Loader + Calibration Ready!")
print("Next step: ORB Feature Detection + Essential Matrix (Monocular VO)")