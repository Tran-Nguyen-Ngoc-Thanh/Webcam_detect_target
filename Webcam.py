import cv2
import numpy as np
import time

# cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

cap = cv2.VideoCapture(2)


cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm độ trễ
cap.set(cv2.CAP_PROP_FPS, 60)  # Tăng FPS để giảm giật lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Đặt độ phân giải
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Đặt độ phân giải

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def capture_frame(camera):
    """ Capture a frame from the video source and convert it to BGR """
    yuv_frame = camera.capture_array()
    return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def calculate_distance(pointcenter, point1):
    """ Calculate the distance between two points """

    # x = -point1[1] + pointcenter[1]
    # y = point1[0] - pointcenter[0]

    x = pointcenter[0] - point1[0]
    y = point1[1] - pointcenter[1]

    return x, y

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def detect_blue_regions(frame, area_threshold=1000):
    # Chuyển đổi từ BGR sang HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)

    hsv = cv2.merge((h, s, v_eq))
    
    lower_blue = np.array([170, 70, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Tạo mặt nạ cho vùng màu xanh
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask_blue_colored = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)

    # Tìm các contour từ mặt nạ
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_circles = []  # Danh sách lưu trữ các vòng tròn màu xanh

    for cnt in contours:
        # Chỉ xử lý những vùng có diện tích đủ lớn
        if cv2.contourArea(cnt) > area_threshold:
            (x, y), radius = cv2.minEnclosingCircle(cnt)  # Tính toán hình tròn bao quanh
            blue_circles.append([x, y, radius])  # Thêm thông tin vòng tròn vào danh sách
            
            cv2.drawContours(mask_blue_colored, [cnt], -1, (255, 0, 0), 2)
            cv2.circle(mask_blue_colored, (int(x), int(y)), int(radius), (255, 0, 0), 2)

 
    if len(blue_circles) > 0:
        blue_circles = np.array(blue_circles, dtype=np.float32)  # Chuyển đổi danh sách thành numpy array
    else:
        blue_circles = None  # Không có vòng tròn nào được phát hiện

    return blue_circles, mask_blue_colored

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def detect_red_regions(frame, area_threshold=1000):
    # Chuyển đổi từ BGR sang HSV    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 90, 90])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 90, 90])
    upper_red2 = np.array([180, 255, 255])

    # Tạo mặt nạ cho vùng màu đỏ
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # Kết hợp hai mặt nạ

    mask_red_colored = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)

    # Tìm các contour từ mặt nạ
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_circles = []  # Danh sách lưu trữ các vòng tròn màu đỏ

    for cnt in contours:
        # Chỉ xử lý những vùng có diện tích đủ lớn
        if cv2.contourArea(cnt) > area_threshold:
            (x, y), radius = cv2.minEnclosingCircle(cnt)  # Tính toán hình tròn bao quanh
            red_circles.append([x, y, radius])  # Thêm thông tin vòng tròn vào danh sách

            cv2.drawContours(mask_red_colored, [cnt], -1, (0, 0, 255), 2)
            cv2.circle(mask_red_colored, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    
    if len(red_circles) > 0:
        red_circles = np.array(red_circles, dtype=np.float32)  # Chuyển đổi danh sách thành numpy array
    else:
        red_circles = None  # Không có vòng tròn nào được phát hiện

    return red_circles, mask_red_colored

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def detect_yellow_regions(frame, area_threshold=1000):
    # Chuyển đổi từ BGR sang HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Định nghĩa ngưỡng cho màu vàng
    lower_yellow = np.array([20, 50, 100])
    upper_yellow = np.array([45, 255, 255])

    # Tạo mặt nạ cho vùng màu vàng
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_yellow_colored = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)

    # Tìm các contour từ mặt nạ
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yellow_circles = []  # Danh sách lưu trữ các vòng tròn màu vàng

    for cnt in contours:
        # Chỉ xử lý những vùng có diện tích đủ lớn
        if cv2.contourArea(cnt) > area_threshold:
            (x, y), radius = cv2.minEnclosingCircle(cnt)  # Tính toán hình tròn bao quanh
            yellow_circles.append([x, y, radius])  # Thêm thông tin vòng tròn vào danh sách
            
            cv2.drawContours(mask_yellow_colored, [cnt], -1, (0, 255, 255), 2)
            cv2.circle(mask_yellow_colored, (int(x), int(y)), int(radius), (0, 255, 255), 2) 


    if len(yellow_circles) > 0:
        yellow_circles = np.array(yellow_circles, dtype=np.float32)  # Chuyển đổi danh sách thành numpy array
    else:

        yellow_circles = None  # Không có vòng tròn nào được phát hiện

    return yellow_circles, mask_yellow_colored

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

# def detect_white_triangles(frame, area_threshold=1000):
#     # Chuyển đổi từ BGR sang HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # Làm đều histogram của kênh sáng (V) để tăng độ tương phản
#     h, s, v = cv2.split(hsv)
#     v_eq = cv2.equalizeHist(v)
#     hsv = cv2.merge((h, s, v_eq))
#     # Định nghĩa khoảng màu trắng
#     lower_white = np.array([0, 0, 170])       # Hue: 0-180, Sat: ~0 (trắng), Value: sáng hơn 170
#     upper_white = np.array([180, 60, 255])    # Sat dưới 60 vẫn coi là trắng (nếu có phản chiếu)
#     # Tạo mặt nạ cho vùng màu trắng
#     mask_white = cv2.inRange(hsv, lower_white, upper_white)
#     # Làm sạch mask nếu cần
#     kernel = np.ones((3, 3), np.uint8)
#     mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
#     mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
#     # Tìm các contour từ mặt nạ
#     contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     white_triangles = []  # Danh sách lưu trữ các tam giác trắng

#     def is_equilateral_triangle(pts, tolerance=0.1):
#         """Kiểm tra xem 3 điểm có tạo thành tam giác đều không (trong sai số cho phép)."""
#         if len(pts) != 3:
#             return False
#         d1 = np.linalg.norm(pts[0][0] - pts[1][0])
#         d2 = np.linalg.norm(pts[1][0] - pts[2][0])
#         d3 = np.linalg.norm(pts[2][0] - pts[0][0])
#         mean_len = (d1 + d2 + d3) / 3.0
#         return all(abs(d - mean_len) / mean_len < tolerance for d in [d1, d2, d3])
    
#     def draw_equilateral_from_triangle(image, triangle_pts, color=(255, 0, 0), thickness=2):
#         """Vẽ một tam giác đều tuyệt đối tại vị trí tam giác gốc, giữ nguyên trọng tâm và chiều dài cạnh trung bình."""
#         pts = triangle_pts.reshape(3, 2)  # Chuyển về 3 điểm (x, y)
        
#         # Tính trọng tâm
#         center = np.mean(pts, axis=0)
#         # Tính chiều dài trung bình các cạnh
#         d1 = np.linalg.norm(pts[0] - pts[1])
#         d2 = np.linalg.norm(pts[1] - pts[2])
#         d3 = np.linalg.norm(pts[2] - pts[0])
#         mean_len = (d1 + d2 + d3) / 3.0
#         # Vẽ tam giác đều dựa trên center và mean_len
#         radius = mean_len / (np.sqrt(3))  # bán kính đường tròn ngoại tiếp
#         new_pts = []
#         for i in range(3):
#             angle = np.radians(i * 120)  # mặc định quay theo trục dọc
#             x = int(center[0] + radius * np.cos(angle))
#             y = int(center[1] + radius * np.sin(angle))
#             new_pts.append((x, y))
        
#         pts_array = np.array(new_pts, dtype=np.int32).reshape((-1, 1, 2))
#         cv2.polylines(image, [pts_array], isClosed=True, color=color, thickness=thickness)
#         return pts_array
#     # Trong vòng lặp xử lý contour:
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > area_threshold:
#             peri = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
#             # Trường hợp có đúng 3 đỉnh
#             if len(approx) == 3:
#                 if is_equilateral_triangle(approx):
#                     white_triangles.append(approx)
#                     cv2.fillPoly(mask_white, [approx], 255)
#                     cv2.polylines(frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)  # Vẽ lên ảnh gốc
#                     # Vẽ lại tam giác đều tuyệt đối
#                     draw_equilateral_from_triangle(frame, approx)
#             # Trường hợp có từ 4-9 đỉnh
#             elif 3 < len(approx) < 10:
#                 hull = cv2.convexHull(approx)
#                 if len(hull) >= 3:
#                     triangle_like = cv2.approxPolyDP(hull, 0.05 * peri, True)
#                     if len(triangle_like) == 3:
#                         if is_equilateral_triangle(triangle_like):
#                             white_triangles.append(triangle_like)
#                             cv2.fillPoly(mask_white, [triangle_like], 255)
#                             # Vẽ lại tam giác đều tuyệt đối
#                             draw_equilateral_from_triangle(mask_white, triangle_like)
#     if len(white_triangles) > 0:
#         white_triangles = np.array(white_triangles)  # Chuyển đổi danh sách thành numpy array
#     else:
#         white_triangles = None  # Không có tam giác nào được phát hiện
#     mask_white_colored = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
#     return white_triangles, mask_white_colored

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def detect_white_triangles(self, frame, area_threshold=1000, min_angle=15, max_angle_deviation=25):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Step 3: Morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    white_triangles = []
    debug_image = frame.copy()

    # Step 5: Loop through contours and look for triangle shapes
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue

        peri = cv2.arcLength(cnt, True)

        # Try multiple epsilon values for polygon approximation
        for epsilon_factor in [0.01, 0.03, 0.05]:
            approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)

            if len(approx) == 3:
                # Validate triangle geometry
                p1, p2, p3 = approx[0][0], approx[1][0], approx[2][0]

                # Side lengths
                a = np.linalg.norm(p2 - p1)
                b = np.linalg.norm(p3 - p2)
                c = np.linalg.norm(p1 - p3)

                # Triangle inequality check
                if a + b <= c or b + c <= a or c + a <= b:
                    continue

                try:
                    # Angles using cosine rule
                    angle_a = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))
                    angle_b = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
                    angle_c = np.degrees(np.arccos((a**2 + b**2 - c**2) / (2 * a * b)))
                    angles = [angle_a, angle_b, angle_c]

                    if min(angles) < min_angle:
                        continue

                    ideal_angle = 60
                    if max(abs(angle - ideal_angle) for angle in angles) > max_angle_deviation:
                        continue

                    # Valid triangle
                    white_triangles.append(approx)
                    cv2.drawContours(debug_image, [approx], 0, (0, 255, 0), 2)
                    break  # Stop trying other epsilons

                except:
                    continue

    # Create mask result
    mask_white_result = np.zeros_like(gray)
    for triangle in white_triangles:
        cv2.fillPoly(mask_white_result, [triangle], 255)

    mask_white_colored = cv2.cvtColor(mask_white_result, cv2.COLOR_GRAY2BGR)

    return white_triangles, mask_white_colored

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def detect_helipad(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    # Apply Gaussian blur to further reduce noise
    filtered = cv2.GaussianBlur(filtered, (5, 5), 0)
    #lighter
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Apply Canny edge detection with optimized thresholds
    edges = cv2.Canny(thresh, 50, 150)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    helipad_candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by area
        if area < 800:  # Adjust threshold as needed
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Aspect ratio check for "H" shape (width to height ratio)
        aspect_ratio = float(w) / h
        if not (0.75 <= aspect_ratio <= 1.25):  # H shape is roughly square
            continue
        
        # Check for "H" pattern using template matching or feature points
        roi = gray[y:y+h, x:x+w]
        
    
    # Draw results
    result = frame.copy()
    for cnt in helipad_candidates:
        cv2.drawContours(result, [cnt], 0, (0, 255, 0), 2)
        
        # Get center point for the helipad
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(result, (cx, cy), 2, (0, 0, 255), -1)
    
    return result, helipad_candidates

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def detect_circles_on_black_background(circles_blue, circles_red, circles_yellow, triangles_white, frame, area_threshold=1000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 155, 100])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morphed = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.dilate(morphed, kernel, iterations=1)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_contour = max(contours, key=cv2.contourArea) if contours else None

    # Tạo mask gốc để vẽ đường viền mà không làm sáng vùng bên trong
    mask_black_area = np.zeros_like(frame)


    if black_contour is not None:
        hull = cv2.convexHull(black_contour)
        epsilon = 0.005 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        cv2.fillPoly(mask_black_area, [approx], (255, 255, 255))

        cv2.drawContours(mask_black_area, [approx], -1, (0, 255, 255), thickness=3)
        cv2.drawContours(frame, [approx], -1, (0, 255, 255), thickness=3)
    else:
        approx = None

    # Lọc vòng tròn nằm trong vùng đen và trong đường viền vàng
    def filter_circles_with_contour(circles, contour, area_threshold=1000):
        valid = []
        if circles is not None and contour is not None:
            for (x, y, r) in circles:
                if np.pi * (r ** 2) > area_threshold:  # Kiểm tra diện tích hình tròn
                    # Kiểm tra tâm hình tròn có nằm trong đường viền không
                    point = (int(x), int(y))
                    if cv2.pointPolygonTest(contour, point, False) >= 0:  # Nằm trong hoặc trên đường viền
                        valid.append([int(x), int(y), int(r)])
        return valid
    
    # Lọc tam giác nằm trong vùng đen và trong đường viền vàng
    def filter_triangles_with_contour(triangles, contour, area_threshold=1000):
        valid = []
        if triangles is not None and contour is not None:
            for tri in triangles:
                area = cv2.contourArea(tri)
                if area > area_threshold:
                    M = cv2.moments(tri)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                            valid.append(tri)
        return valid
    
   

    if approx is not None:
        valid_blue = filter_circles_with_contour(circles_blue, approx, area_threshold)
        valid_red = filter_circles_with_contour(circles_red, approx, area_threshold)
        valid_yellow = filter_circles_with_contour(circles_yellow, approx, area_threshold)

        valid_white_triangles = filter_triangles_with_contour(triangles_white, approx, area_threshold)
    else:
        valid_blue, valid_red, valid_yellow, valid_white_triangles = None, None, None, None


    return (
        np.array(valid_blue, dtype=np.float32) if valid_blue else None,
        np.array(valid_red, dtype=np.float32) if valid_red else None,
        np.array(valid_yellow, dtype=np.float32) if valid_yellow else None,
        valid_white_triangles if valid_white_triangles else None,
        mask_black_area, frame   # Chỉ có đường viền vàng, không tô vùng bên trong
    )





#---------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def visualize(image, text, unit, row_size, color, numberdis: float) -> np.ndarray:
    """ Overlay the number_dis value onto the given image. """
    if color == 'green':
        color = (0, 255, 0)
    elif color == 'brown':
        color = (42, 42, 165)
    elif color == 'pink':
        color = (255, 0, 255)
    elif color == 'blue':
        color = (255, 0, 0)
    elif color == 'red':
        color = (0, 0, 255)
    else:
        color = (255, 255, 255)

    left_margin = 24  # pixels
    font_size = 1
    font_thickness = 1

    numberdis_text = text + ': {:.1f}'.format(numberdis) + unit

    text_location = (left_margin, row_size)
    cv2.putText(image, numberdis_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, color, font_thickness)

    return image

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def get_color_name(r, g, b):
    """ Returns the accurate color name based on RGB values by converting to the HSV """
    color_bgr = np.uint8([[[b, g, r]]])  # OpenCV dùng BGR
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv  # Hue, Saturation, Value

    # Xác định màu theo độ sáng trước
    if v < 50:
        return "Black"
    elif v > 200 and s < 50:
        return "White"
    elif s < 30:
        return "Gray"

    # Xác định màu dựa trên Hue
    if (0 <= h < 5) or (h >= 175):
        return "Red"
    elif 20 <= h < 50:
        return "Yellow"
    elif 85 <= h < 120:
        return "Cyan"
    elif 120 <= h < 175:
        return "Blue"

    return "Unknown"

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def get_dominant_color_name(roi):
    """ Return the most dominant color name in the ROI area """
    roi_small = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
    pixels = roi_small.reshape(-1, 3)
    avg_color = np.mean(pixels, axis=0)
    r, g, b = int(avg_color[2]), int(avg_color[1]), int(avg_color[0])
    return get_color_name(r, g, b), (b, g, r)

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def process_frame(frame_o):
    """ Blur, convert to grayscale and detect circles """
    frame = cv2.blur(frame_o, (3, 3))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blue_circles, mask_blue = detect_blue_regions(frame)
    red_circles, mask_red = detect_red_regions(frame)
    yellow_circles, mask_yellow = detect_yellow_regions(frame)
    white_triangles, mask_white_tri = detect_white_triangles(frame)


    blue_on_black, red_on_black, yellow_on_black, white_triangles_black, mask_black, output = detect_circles_on_black_background(
        blue_circles, red_circles, yellow_circles, white_triangles, frame)


    return frame, mask_blue, mask_red, mask_yellow, mask_white_tri, mask_black, blue_on_black, red_on_black, yellow_on_black, white_triangles_black, output
    #return frame, mask_blue, mask_red, mask_yellow, blue_circles, red_circles, yellow_circles, output

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def draw_shapes_on_frame(frame, blue_circles=None, red_circles=None, yellow_circles=None, white_triangles=None):
    """ Draw the largest circle and display the dominant color """
    output = frame.copy()
    center_x = output.shape[1] // 2
    center_y = output.shape[0] // 2
    cv2.circle(output, (center_x, center_y), 4, (255, 255, 200), -1)  # Tâm khung ảnh

    all_circles = []  # Danh sách lưu trữ tất cả các vòng tròn,
    all_triangles = []  # Danh sách lưu trữ tất cả các tam giác;
    all_squares = []  # Danh sách lưu trữ tất cả các hình vuông;

    # Thêm vòng tròn từ Hough Circle vào danh sách
    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #     all_circles.extend(circles)

    # Thêm vòng tròn từ vùng màu xanh vào danh sách
    if blue_circles is not None:
        blue_circles = np.round(blue_circles).astype("int")  # Đảm bảo blue_circles cũng là kiểu int
        all_circles.extend(blue_circles)

    # Thêm vòng tròn từ vùng màu đỏ vào danh sách
    if red_circles is not None:
        red_circles = np.round(red_circles).astype("int")  # Đảm bảo red_circles cũng là kiểu int
        all_circles.extend(red_circles)

    # Thêm vòng tròn từ vùng màu vàng vào danh sách
    if yellow_circles is not None:
        yellow_circles = np.round(yellow_circles).astype("int")  # Đảm bảo yellow_circles cũng là kiểu int
        all_circles.extend(yellow_circles)

    # Thêm tam giác màu trắng vào danh sách
    if white_triangles is not None:
        all_triangles.extend(white_triangles)

    # Tìm hình tròn lớn nhất
    largest_circle = None
    largest_radius = 0
    largest_triangle = None
    larget_area_tri = 0
    
    for (x, y, r) in all_circles:
        if r > largest_radius:
            largest_radius = r
            largest_circle = (x, y, r)


    for triangle in all_triangles:
        area = cv2.contourArea(np.array(triangle))
        if area > larget_area_tri:
            larget_area_tri = area
            largest_triangle = triangle
    

    # Vẽ hình tròn lớn nhất nếu tìm thấy
    if largest_circle is not None:
        x, y, r = largest_circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)  # Vẽ hình tròn lớn nhất màu xanh lá
        cv2.circle(output, (x, y), 4, (0, 0, 255), -1)  # Vẽ tâm hình tròn màu đỏ

        # Vẽ các đường chỉ hướng
        cv2.line(output, (center_x, center_y), (x, y), (0, 0, 0), 2)  # Đường thẳng màu hồng
        cv2.line(output, (center_x, center_y), (center_x, y), (42, 42, 165), 2)  # Đường dọc màu nâu
        cv2.line(output, (x, y), (center_x, y), (255, 0, 0), 2)  # Đường ngang màu xanh dương
        cv2.circle(output, (center_x, y), 4, (0, 0, 0), -1)  # Vẽ điểm giao nhau màu đen

        # Tính khoảng cách
        x11, y11 = calculate_distance((center_x, center_y), (x, y))

        # Hiển thị khoảng cách
        visualize(output, 'x', 'px', 40, 'pink', x11)
        visualize(output, 'y', 'px', 60, 'pink', y11)

        # Tạo mask và lấy ROI để phân tích màu
        mask = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        masked_frame = cv2.bitwise_and(output, output, mask=mask)
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(output.shape[1], x + r), min(output.shape[0], y + r)
        roi = masked_frame[y1:y2, x1:x2]

        color_name, dominant_bgr = get_dominant_color_name(roi)

        # Hiển thị tên màu
        cv2.putText(output, f"Circle - {color_name}", (x - 40, y - r - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, dominant_bgr, 2)
    
    if largest_triangle is not None:
        triangle = np.array(largest_triangle).reshape((-1, 1, 2))
        cv2.drawContours(output, [triangle], -1, (255, 255, 255), 3)

        M = cv2.moments(triangle)
        if M["m00"] != 0:  # Kiểm tra diện tích khác 0 để tránh chia cho 0
            cx = int(M["m10"] / M["m00"])  # Tính tọa độ x của trọng tâm
            cy = int(M["m01"] / M["m00"])  # Tính tọa độ y của trọng tâm

            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)  # Vẽ tâm tam giác (trọng tâm)
            cv2.line(output, (center_x, center_y), (cx, cy), (0, 0, 0), 2)  # Đoạn thẳng giữa tâm ảnh và trọng tâm tam giác
            cv2.line(output, (center_x, center_y), (center_x, cy), (42, 42, 165), 2)  # Đoạn dọc
            cv2.line(output, (cx, cy), (center_x, cy), (255, 0, 0), 2)  # Đoạn ngang
            cv2.circle(output, (center_x, cy), 4, (0, 0, 0), -1)  # Điểm giao nhau

            dx, dy = calculate_distance((center_x, center_y), (cx, cy))  # Tính khoảng cách
            visualize(output, 'x', 'px', 40, 'white', dx)  # Hiển thị khoảng cách x
            visualize(output, 'y', 'px', 60, 'white', dy)  # Hiển thị khoảng cách y

            # Tạo mask cho tam giác
            mask_triangle = np.zeros_like(output, dtype=np.uint8)  # Khởi tạo mask đen
            cv2.fillPoly(mask_triangle, [triangle], (255))  # Điền màu trắng vào vùng tam giác

            # Áp dụng mask lên ảnh gốc để chỉ giữ lại vùng tam giác
            masked_frame_triangle = cv2.bitwise_and(output, mask_triangle)

            # Cắt ROI từ vùng tam giác (tùy chọn, nếu muốn cắt vùng nhỏ)
            roi_triangle = masked_frame_triangle  # Chỉ cần ROI chính là vùng đã được mask

            # Phân tích màu sắc trong tam giác
            color_name, dominant_bgr = get_dominant_color_name(roi_triangle)  # Phân tích màu sắc trong ROI tam giác
            cv2.putText(output, f"Triangle - {color_name}", (cx - 40, cy - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, dominant_bgr, 2)  # Hiển thị tên màu sắc
   
   
    return output

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def anisotropic_diffusion(img, num_iter=3, kappa=30, gamma=0.1):
    """
    Hàm thực hiện Anisotropic Diffusion trên hình ảnh.

    Parameters:
        img (ndarray): Hình ảnh đầu vào.
        num_iter (int): Số lần lặp.
        kappa (float): Tham số điều chỉnh độ nhạy với độ sáng.
        gamma (float): Tham số điều chỉnh độ lớn của bước.

    Returns:
        output (ndarray): Hình ảnh đã được xử lý.
    """
    img = img.astype(np.float32) / 255.0  # Chuyển đổi sang float
    for _ in range(num_iter):
        # Tính toán gradient
        dx = np.roll(img, -1, axis=1) - img
        dy = np.roll(img, -1, axis=0) - img
        # Tính toán tính toán nhịp
        c = np.exp(-(dx ** 2 + dy ** 2) / kappa)
        # Cập nhật hình ảnh
        img += gamma * (c * (dx + dy))

    return (img * 255).astype(np.uint8)

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def combine_images_grid(img_grid, titles=None, border_thickness=2, border_color=(0, 255, 0)):
    
    # Xác định kích thước chuẩn
    h, w = img_grid[0][0].shape[:2]
    dtype = img_grid[0][0].dtype

    combined_rows = []

    for row_idx, row in enumerate(img_grid):
        processed_row = []
        for col_idx, img in enumerate(row):
            # Chuyển sang ảnh màu nếu cần
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Resize ảnh về kích thước chuẩn
            img = cv2.resize(img, (w, h))

            # Ép kiểu dữ liệu nếu khác
            if img.dtype != dtype:
                img = img.astype(dtype)

            # Thêm viền quanh ảnh
            img = cv2.copyMakeBorder(img, border_thickness, border_thickness,
                                     border_thickness, border_thickness,
                                     cv2.BORDER_CONSTANT, value=border_color)

            # Thêm tiêu đề nếu có
            if titles:
                title = titles[row_idx][col_idx]
                cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)

            processed_row.append(img)

        # Ghép các ảnh trong hàng
        row_img = cv2.hconcat(processed_row)
        combined_rows.append(row_img)

    # Cuối cùng, ghép các hàng lại theo chiều dọc
    combined_image = cv2.vconcat(combined_rows)
    return combined_image

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

def main():
    try:
        # Kiểm tra xem camera có mở được không
        if not cap.isOpened():
            print("Không thể mở webcam ngoài! Hãy kiểm tra lại ID.")
            exit()

        while True:
            start_time = time.time()

            # Initialize Picamera2
            ret, frame = cap.read()
            if not ret:
                break


            # Resize ảnh xuống 50% để tối ưu xử lý
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)


            # Giảm chói bằng anisotropic diffusion (giữ lại màu sắc)
            processed_color_frame = anisotropic_diffusion(frame)

            # Tiếp tục xử lý để phát hiện hình tròn, tam giác (lưu ý: process_frame chuyển sang grayscale nên hãy sử dụng ảnh đã giảm chói để phát hiện các hình dạng)

            blur_frame, mask_blue, mask_red, mask_yellow, mask_white_tri, mask_black, blue_on_black, red_on_black, yellow_on_black, white_triangles_black, Output = process_frame(processed_color_frame)
  


            Output = draw_shapes_on_frame(blur_frame, blue_on_black, red_on_black, yellow_on_black, white_triangles_black)
   
            end_time = time.time()
            seconds = end_time - start_time
            fps = 1.0 / seconds

            # Gắn thông tin FPS lên ảnh output
            Out_final = visualize(Output, 'FPS', 'hz', 20, 'green', fps)

            # Kết hợp các ảnh để hiển thị cùng 1 cửa sổ: ảnh gốc, ảnh vùng màu xanh, và ảnh output có vẽ các hình
            img_list = [[Out_final,mask_black, mask_blue], [mask_red, mask_yellow, mask_white_tri]]

            titles = [["hinh vuong","black mask","blue Mask"], ["red mask", "yellow mask", "tam giac"]]
            mask = combine_images_grid(img_list, titles)

            cv2.imshow("Combined", mask)      


            if cv2.waitKey(33) == 27:  # Escape key
                break

    except Exception as e:
        print(e)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
