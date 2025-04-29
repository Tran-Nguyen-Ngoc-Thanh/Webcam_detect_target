import cv2
import numpy as np
import os

def is_equilateral_triangle(pts, tolerance=0.2):
    a = np.linalg.norm(pts[0][0] - pts[1][0])
    b = np.linalg.norm(pts[1][0] - pts[2][0])
    c = np.linalg.norm(pts[2][0] - pts[0][0])

    max_len = max(a, b, c)
    min_len = min(a, b, c)
    return (max_len - min_len) / max_len < tolerance

# Đọc ảnh
folder_path = r"D:\Documents\OpencvApp\image\center_cam"
output_path = os.path.join(folder_path, "output_with_centers")
os.makedirs(output_path, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Không đọc được ảnh: {filename}")
            continue

        orig = img.copy()

        # === Giảm nếp nhăn ===
        denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated_mask = cv2.dilate(white_mask, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)

        white_mask_edge = cv2.Canny(eroded_mask, 40, 150)
        cv2.imshow("White Mask Edge", white_mask_edge)

        contours, _ = cv2.findContours(white_mask_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow("Contours", cv2.drawContours(orig.copy(), contours, -1, (0, 255, 0), 3))

        # Lọc các contour có hình tam giác đều
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)

            if len(approx) == 3 and area > 500 and area < 2000:
                if is_equilateral_triangle(approx):
                    cv2.drawContours(orig, [approx], -1, (0, 255, 0), 3)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.putText(orig, "Equilateral Triangle", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Hiển thị kết quả
        cv2.imshow("Detected Helipad Triangle", orig)
        cv2.waitKey(0)

cv2.destroyAllWindows()
