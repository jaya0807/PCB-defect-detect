"""
Downloader for real PCB Defect Images from the PKU-PCB benchmark dataset
(The exact dataset used in the Kaggle notebook 'pcb-defect-detection-with-yolov8')
"""

import os
import sys
import json
import urllib.request

CATEGORIES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]

TARGET_DIRS = [
    r"C:\Users\USER\Desktop\fault pcb image",
    r"C:\Users\USER\.gemini\antigravity\scratch\pcb-defect-ai\test_defect_images",
]

# Number of sample images per category to download
IMAGES_PER_CATEGORY = 6


def download_dataset():
    for target in TARGET_DIRS:
        os.makedirs(target, exist_ok=True)
        print(f"Target directory ready: {target}")

    headers = {"User-Agent": "Mozilla/5.0"}
    total_downloaded = 0

    for cat in CATEGORIES:
        print(f"\nFetching image list for: {cat}...")
        api_url = f"https://api.github.com/repos/Ironbrotherstyle/PCB-DATASET/contents/images/{cat}"
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req) as resp:
                items = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Error fetching directory {cat}: {e}")
            continue

        # Filter for .jpg files
        jpg_items = [it for it in items if it["name"].lower().endswith(".jpg")]
        selected = jpg_items[:IMAGES_PER_CATEGORY]

        # Create category subdirectories on Desktop
        for target in TARGET_DIRS:
            cat_dir = os.path.join(target, cat)
            os.makedirs(cat_dir, exist_ok=True)

        for item in selected:
            fname = item["name"]
            raw_url = f"https://raw.githubusercontent.com/Ironbrotherstyle/PCB-DATASET/master/images/{cat}/{fname}"
            try:
                raw_req = urllib.request.Request(raw_url, headers=headers)
                with urllib.request.urlopen(raw_req) as img_resp:
                    img_data = img_resp.read()

                # Save to both Desktop and workspace
                for target in TARGET_DIRS:
                    out_path = os.path.join(target, cat, fname)
                    with open(out_path, "wb") as f:
                        f.write(img_data)

                total_downloaded += 1
                print(f"  [OK] Downloaded {cat}/{fname} ({len(img_data) // 1024} KB)")
            except Exception as err:
                print(f"  [FAIL] Failed to download {fname}: {err}")

    print(f"\n==========================================")
    print(f"Successfully downloaded {total_downloaded} defect PCB images!")
    print(f"Saved to: C:\\Users\\USER\\Desktop\\fault pcb image")
    print(f"Also mirrored to: C:\\Users\\USER\\.gemini\\antigravity\\scratch\\pcb-defect-ai\\test_defect_images")
    print(f"==========================================")


if __name__ == "__main__":
    download_dataset()

