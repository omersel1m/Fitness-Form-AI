import cv2
import numpy as np
from ultralytics import YOLO
import time

# Model yolu
model_path = "yolov8n-pose.pt"

# Modeli yükle
model = YOLO(model_path)

# Kameradan görüntü yakalama
cap = cv2.VideoCapture(0)

elbow_positions = []
hip_positions = []
start_position = None
tolerance = 50

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Modeli görüntü üzerinde çalıştır ve sonuçları al
    results = model(frame)[0]

    # Keypoint ve bağlantıların otomatik çizimi
    plotted_img = results.plot()

    # Görüntü üzerine keypoint numaralarını ekle
    if results.keypoints is not None:
        
        keypoints = results.keypoints.xy[0].cpu().numpy()  # Keypoint noktalarını numpy array'e çevir
        yuk, gen, _ = frame.shape
        
        # Keypoint'leri görselleştir
        for kp_id, (x, y) in enumerate(keypoints):
            x, y = int(x), int(y)  # Normalize edilmemiş şekilde kullanmak
            cv2.circle(plotted_img, (x, y), 5, (0, 255, 0), -1)  # Keypoint'leri göster
            cv2.putText(plotted_img, str(kp_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        left_elbow_pos = (keypoints[7][0], keypoints[7][1])
        right_elbow_pos = (keypoints[8][0], keypoints[8][1])
        left_hip_pos = (keypoints[11][0], keypoints[11][1])
        right_hip_pos = (keypoints[12][0] , keypoints[12][1])
        
        # 7. saniyedeki pozisyonları başlangıç noktası olarak belirleme
        elapsed_time = time.time() - start_time
        if elapsed_time >= 7     and start_position is None:
            start_position = {
                'left_elbow': left_elbow_pos,
                'right_elbow': right_elbow_pos,
                'left_hip': left_hip_pos,
                'right_hip': right_hip_pos
            }
            print("Start positions set:", start_position)
        
        # Başlangıç pozisyonları belirlendikten sonra işlemlere devam etme
        if start_position:
            # Dirsek ve bel pozisyonlarını listeye ekleme
            elbow_positions.append(left_elbow_pos)
            hip_positions.append(left_hip_pos)
            
            if len(elbow_positions) > 100:
                elbow_positions.pop(0)
            
            if len(hip_positions) > 100:
                hip_positions.pop(0)
            
            # Son 100 pozisyondan ikişer ikişer seçerek varyans hesaplama
            selected_elbow_positions = elbow_positions[::2]
            selected_hip_positions = hip_positions[::2]
            
            # Dirsek pozisyonları için varyans hesaplama
            elbow_positions_array = np.array(selected_elbow_positions)
            elbow_variance = np.var(elbow_positions_array, axis=0)
            
            # Bel pozisyonları için varyans hesaplama
            hip_positions_array = np.array(selected_hip_positions)
            hip_variance = np.var(hip_positions_array, axis=0)
            
            # Tolerans kontrolü
            left_elbow_diff = np.linalg.norm(np.array(left_elbow_pos) - np.array(start_position['left_elbow']))
            right_elbow_diff = np.linalg.norm(np.array(right_elbow_pos) - np.array(start_position['right_elbow']))
            left_hip_diff = np.linalg.norm(np.array(left_hip_pos) - np.array(start_position['left_hip']))
            right_hip_diff = np.linalg.norm(np.array(right_hip_pos) - np.array(start_position['right_hip']))
            
            # Toplam sapmayı hesapla
            total_diff = (left_elbow_diff + right_elbow_diff + left_hip_diff + right_hip_diff) / 4
            
            # Maksimum sapmayı hesaba katarak doğruluk yüzdesi hesapla
            max_diff = tolerance * 4
            accuracy_percentage = max(0, 100 - (total_diff / max_diff) * 100)
            
            cv2.putText(plotted_img, f"Hareketin dogruluk yuzdesi: {accuracy_percentage:.2f}%", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
    # Görüntüyü göster
    cv2.imshow('Pose Estimation', plotted_img)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
