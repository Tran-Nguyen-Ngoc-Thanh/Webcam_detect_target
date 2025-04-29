import cv2
import numpy as np
import os

#folder_path = r"C:\Users\QK\Desktop\UAVcontest\UAVcontest_1\yellow_12h_16.4"
#folder_path = r"C:\Users\QK\Desktop\UAVcontest\yellow_image_16.4"
#folder_path = r"C:\Users\QK\Desktop\UAVcontest\Center_11_45"
#folder_path = r"C:\Users\QK\Desktop\UAVcontest\center_cam"
#folder_path = r"C:\Users\QK\Desktop\UAVcontest\heliport"
#folder_path = r"C:\Users\QK\Desktop\UAVcontest\hehe"
folder_path = r"C:\Users\QK\Desktop\UAVcontest\test_22.4"
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
        #img = cv2.resize(img, (640, 480))

        if img is None:
            print(f"Không đọc được ảnh: {filename}")
            continue

        img = simple_white_balance(img)
        #img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Tạo mask cho từng màu
        yellow_mask = cv2.inRange(hsv, (10, 50, 100), (60, 255, 255))

        lower_red1 = (0, 100, 90)
        upper_red1 = (8, 255, 255)
        lower_red2 = (165, 100, 90)
        upper_red2 = (180, 255, 255)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        blue_mask = cv2.inRange(hsv, (90, 100, 220), (150, 255, 255))

        # Danh sách các mask và màu vẽ tương ứng
        masks = [("Yellow", yellow_mask, (255, 255, 255)), ("Red", red_mask, (255, 255, 255)), ("Blue", blue_mask, (255, 255, 255))]

        img_result = img.copy()

        for color_name, mask, draw_color in masks:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.imshow(f"{color_name} Mask", mask)

            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                # Phát hiện hình gần tròn bằng fitEllipse
                if area > 1000 and area < 20000 and len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (major, minor) = ellipse[1]
                    if 0.75 <= minor / major <= 1.25:
                        cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
                        cv2.ellipse(img_result, ellipse, draw_color, 2)
                        cv2.circle(img_result, (cx, cy), 5, (0, 255, 0), -1)
                        cv2.putText(img_result, f"{color_name}: ({cx},{cy})", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
                        print(f"{filename} - {color_name} center: ({cx}, {cy})")

        cv2.imshow("Result", img_result)
        cv2.waitKey(0)

        save_path = os.path.join(output_path, f"centers_{filename}")
        cv2.imwrite(save_path, img_result)

cv2.destroyAllWindows()
