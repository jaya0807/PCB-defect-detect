"""
Reference-Image Comparison & Homography Alignment Module
Compares an inspected PCB against a 'Golden Master' CAD or reference board.
Performs feature-based registration, absolute difference extraction, and discrepancy heatmaps.
"""

from typing import Tuple, Dict, Any, List
import cv2
import numpy as np


class PCBComparator:
    def __init__(self, n_features: int = 2000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def align_and_compare(
        self,
        reference_img: np.ndarray,
        test_img: np.ndarray,
        diff_threshold: int = 35,
        min_discrepancy_area: int = 50
    ) -> Dict[str, Any]:
        """
        Aligns test_img to reference_img using ORB and Homography,
        computes pixel-level difference, and produces a colored anomaly heatmap.
        
        Args:
            reference_img: Golden PCB image (RGB or BGR numpy array)
            test_img: Tested/Manufactured PCB image
            diff_threshold: Pixel intensity threshold to register as defect
            min_discrepancy_area: Minimum pixel area to consider a valid anomaly
        """
        # Ensure BGR format for OpenCV operations
        ref_bgr = self._to_bgr(reference_img)
        test_bgr = self._to_bgr(test_img)

        # Convert to grayscale for alignment
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(test_gray, None)

        alignment_success = False
        aligned_test_bgr = test_bgr

        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
            # KNN match with Lowe's ratio test
            matches = self.matcher.knnMatch(des2, des1, k=2)
            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= 8:
                src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    h, w = ref_bgr.shape[:2]
                    aligned_test_bgr = cv2.warpPerspective(test_bgr, H, (w, h))
                    alignment_success = True

        # If homography failed or shapes differ, resize test image to match reference
        if not alignment_success or aligned_test_bgr.shape[:2] != ref_bgr.shape[:2]:
            aligned_test_bgr = cv2.resize(test_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))

        # Compute Difference & Heatmap
        aligned_gray = cv2.cvtColor(aligned_test_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(ref_gray, aligned_gray)

        # Apply Gaussian blur to reduce high-frequency noise
        diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(diff_blur, diff_threshold, 255, cv2.THRESH_BINARY)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh_clean = cv2.dilate(thresh_clean, kernel, iterations=2)

        # Find discrepancy contours
        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        discrepancies: List[Dict[str, Any]] = []

        highlighted_test = aligned_test_bgr.copy()
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area >= min_discrepancy_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                discrepancies.append({
                    "id": idx + 1,
                    "bbox": [x, y, x + w, y + h],
                    "center_px": (cx, cy),
                    "area_px": int(area),
                })
                # Highlight in Cyan/Red on test board
                cv2.rectangle(highlighted_test, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    highlighted_test,
                    f"DIFF #{len(discrepancies)}",
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

        # Difference Heatmap
        heatmap_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(aligned_test_bgr, 0.6, heatmap_color, 0.4, 0)

        # Convert outputs back to RGB for display
        return {
            "alignment_success": alignment_success,
            "discrepancies_count": len(discrepancies),
            "discrepancies": discrepancies,
            "reference_rgb": cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB),
            "aligned_test_rgb": cv2.cvtColor(aligned_test_bgr, cv2.COLOR_BGR2RGB),
            "diff_mask_rgb": cv2.cvtColor(thresh_clean, cv2.COLOR_GRAY2RGB),
            "heatmap_rgb": cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB),
            "highlighted_test_rgb": cv2.cvtColor(highlighted_test, cv2.COLOR_BGR2RGB),
        }

    def _to_bgr(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:
            # Assume input from PIL/Streamlit is RGB
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

