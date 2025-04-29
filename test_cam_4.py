import cv2
import numpy as np
import os

folder_path = r"D:\Documents\OpencvApp\image\center_cam"
#folder_path = r"C:\Users\QK\Desktop\UAVcontest\yellow_image_16.4"
output_path = os.path.join(folder_path, "output_with_centers")
os.makedirs(output_path, exist_ok=True)

def simple_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.5)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.5)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Không đọc được ảnh: {filename}")
            continue

        img = simple_white_balance(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Tạo mask cho từng màu
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))

        lower_red1 = (0, 100, 90)
        upper_red1 = (20, 255, 255)
        lower_red2 = (340, 100, 90)
        upper_red2 = (360, 255, 255)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        blue_mask = cv2.inRange(hsv, (90, 100, 80), (150, 255, 255))

        # Danh sách các mask và màu vẽ tương ứng
        masks = [("Yellow", yellow_mask, (255, 255, 255)), ("Red", red_mask, (255, 255, 255)), ("Blue", blue_mask, (255, 255, 255))]

        img_result = img.copy()

        for color_name, mask, draw_color in masks:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                # Tính chỉ số độ tròn
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Giữ lại nếu là hình gần tròn và đủ lớn
                if 0.7 < circularity < 1.3 and area > 1000:
                    cv2.drawContours(img_result, [contour], -1, draw_color, 2)

                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(img_result, (cx, cy), 5, (0, 255, 0), -1)
                        cv2.putText(img_result, f"{color_name}: ({cx},{cy})", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
                        print(f"{filename} - {color_name} center: ({cx}, {cy})")

        cv2.imshow("Result", img_result)
        cv2.waitKey(0)

        save_path = os.path.join(output_path, f"centers_{filename}")
        cv2.imwrite(save_path, img_result)

cv2.destroyAllWindows()
