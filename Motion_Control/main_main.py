import cv2
from ultralytics import YOLO
import time
import math
from pymavlink import mavutil

# Menghubungkan ke flight controller melalui port USB/serial
print("Connecting to vehicle...")
connection = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
connection.wait_heartbeat()
print("Connected to vehicle!")

# Override nilai RC1-RC4
rc1_value_awal = 1500  # Nilai untuk RC channel 1 (range: 1000-2000)
rc2_value_awal = 1500  # Nilai untuk RC channel 2 (range: 1000-2000)
rc3_value_awal = 1000  # Nilai untuk RC channel 3 (range: 1000-2000)
rc4_value_awal = 1500  # Nilai untuk RC channel 4 (range: 1000-2000)

# lurus#
rc1_value_lurus = 1500  # Nilai untuk RC channel 2 (range: 1000-2000)
rc3_value_lurus = 1500

# kiri mentok#
rc1_value_kiri_mentok = 1000  # Nilai untuk RC channel 2 (range: 1000-2000)

# kanan mentok#
rc1_value_kanan_mentok = 2000  # Nilai untuk RC channel 2 (range: 1000-2000)


# kiri sitik
rc1_value_kiri_sitik = 1300

# kanan sitik
rc1_value_kanan_sitik = 1800

# Fungsi untuk mendapatkan data dari vehicle (flight controller)


def override_rc_channels(connection, rc1_value, rc2_value, rc3_value, rc4_value):
    # Mengirimkan perintah COMMAND_LONG untuk override channel RC1-RC4
    try:
        connection.mav.rc_channels_override_send(
            # target_system (ID dari flight controller)
            connection.target_system,
            # target_component (biasanya ID dari flight controller)
            connection.target_component,
            rc1_value,  # Nilai override untuk RC channel 1 (1000-2000)
            rc2_value,  # Nilai override untuk RC channel 2 (1000-2000)
            rc3_value,  # Nilai override untuk RC channel 3 (1000-2000)
            rc4_value,  # Nilai override untuk RC channel 4 (1000-2000)
            # Set remaining channels (RC5 to RC8) to 0 (no override)
            0, 0, 0, 0
        )
    except Exception as e:
        print(f"Failed to override RC channels: {e}")


def get_vehicle_data():
    # Menerima pesan MAVLink
    msg_gps = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    msg_att = connection.recv_match(type='ATTITUDE', blocking=True)
    msg_bat = connection.recv_match(type='SYS_STATUS', blocking=True)
    msg_vfr = connection.recv_match(type='VFR_HUD', blocking=True)
    msg_rc = connection.recv_match(type='RC_CHANNELS', blocking=True)

    # Coba untuk menerima pesan SERVO_OUTPUT_RAW, tapi tidak wajib blocking=True
    msg_servo = connection.recv_match(type='SERVO_OUTPUT_RAW', blocking=False)

    # Mengambil data GPS (lokasi, ketinggian, dan ground course)
    gps = {
        'lat': msg_gps.lat / 1e7,   # Latitude
        'lon': msg_gps.lon / 1e7,   # Longitude
        'alt': msg_gps.alt / 1e3,   # Altitude (meters)
        'cog': msg_gps.hdg / 100.0  # Course over ground (degrees)
    }

    # Mengambil data status baterai
    batt_status = {
        'voltage': msg_bat.voltage_battery / 1000.0,  # Tegangan baterai dalam volt
        'current': msg_bat.current_battery / 100.0,   # Arus baterai dalam Ampere
        'level': msg_bat.battery_remaining            # Level baterai dalam %
    }

    # Mengambil data attitude (roll, pitch, yaw)
    attitude = {
        'roll': math.degrees(msg_att.roll),
        'pitch': math.degrees(msg_att.pitch),
        'yaw': math.degrees(msg_att.yaw)
    }

    # Mengambil data kecepatan tanah dalam beberapa satuan
    speed = {
        'ground_speed': msg_vfr.groundspeed,    # Groundspeed dalam m/s
        'kmh': msg_vfr.groundspeed * 3.6,       # Groundspeed dalam km/h
        'knot': msg_vfr.groundspeed * 1.94384   # Groundspeed dalam knot
    }

    # Mengambil data tambahan lainnya
    heading = msg_vfr.heading  # Compass heading
    baro = msg_vfr.alt  # Altitude relatif dari barometer

    # Mengambil data RC channel
    rc_channels = {
        'rc1': msg_rc.chan1_raw,
        'rc2': msg_rc.chan2_raw,
        'rc3': msg_rc.chan3_raw,
        'rc4': msg_rc.chan4_raw,
        'rc5': msg_rc.chan5_raw,
        'rc6': msg_rc.chan6_raw
    }

    # Mengambil nilai servo output jika tersedia
    if msg_servo:
        servo_output = {
            'steer': msg_servo.servo1_raw,  # Servo 1
            'th_mid': msg_servo.servo3_raw,  # Servo 3
            'th_left': msg_servo.servo5_raw,  # Servo 5
            'th_right': msg_servo.servo7_raw  # Servo 7
        }
    else:
        # Jika pesan servo tidak ada, set sebagai None atau nilai default
        servo_output = {
            'steer': None,
            'th_mid': None,
            'th_left': None,
            'th_right': None
        }

    # Mendapatkan mode kendaraan
    msg_heartbeat = connection.recv_match(type='HEARTBEAT', blocking=True)
    mode = mavutil.mode_string_v10(msg_heartbeat)

    # Memeriksa apakah kendaraan armed
    is_armed = msg_heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED > 0

    # Memeriksa apakah kendaraan dapat di-arm
    is_armable = (
        mode in ["GUIDED", "AUTO", "STABILIZE"] and
        batt_status['level'] > 20 and  # Level baterai lebih dari 20%
        msg_bat.current_battery >= 0  # Status sistem tidak menunjukkan masalah
    )

    # Mengembalikan semua data dalam satu dictionary
    return {
        "gps": gps,
        "bat_status": batt_status,
        "heading": heading,
        "speed": speed,
        "baro": baro,
        "attitude": attitude,
        "rc_channels": rc_channels,  # Data RC
        "servo_output": servo_output,  # Data Servo
        "mode": mode,                  # Mode kendaraan
        "is_armed": is_armed,           # Status armed
        "is_armable": is_armable,        # Status armable
    }


# Konstanta
MODEL_PATH = '../models/best.egine'
# VIDEO_PATH = 0
VIDEO_PATH = "../B.mp4"
FRAME_WIDTH, FRAME_HEIGHT = 640, 640

# Dictionary untuk mapping class ID ke nama
class_names = {0: 'green_ball', 1: 'red_ball'}

# List untuk menyimpan warna berdasarkan class ID
class_colors = [
    (0, 255, 0),  # Hijau untuk green_ball (ID 0)
    (0, 0, 255)   # Merah untuk red_ball (ID 1)
]

# Hitung titik tengah frame
center_x_frame = FRAME_WIDTH // 2
left_x_frame = 100
right_x_frame = 540

# Titik acuan pivot
pivot_point = (FRAME_WIDTH // 2, FRAME_HEIGHT)
bottom_pivot = 340
fps = 0.00


def draw_center_line(frame, fps, green, red, mid):
    """
    Fungsi untuk menggambar garis tengah berwarna putih pada frame.
    """
    # Tampilkan FPS di frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if green is not None:
        green_center_x, green_center_y = green[4]
        cv2.putText(frame, f'Green: ({green_center_x:.0f}, {green_center_y:.0f})', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if red is not None:
        red_center_x, red_center_y = red[4]
        cv2.putText(frame, f'Red: ({red_center_x:.0f}, {red_center_y:.0f})', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, f'Mid: ({mid[0]:.0f}, {mid[1]:.0f})', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Garis vertikal di tengah frame
    cv2.line(frame, (center_x_frame, 0),
             (center_x_frame, FRAME_HEIGHT), (255, 255, 255), 2)

    # Garis vertikal di kiri frame
    cv2.line(frame, (left_x_frame, 0),
             (left_x_frame, FRAME_HEIGHT), (255, 255, 255), 2)

    # Garis vertikal di kanan frame
    cv2.line(frame, (right_x_frame, 0),
             (right_x_frame, FRAME_HEIGHT), (255, 255, 255), 2)

    # Garis horizontal di kanan frame
    cv2.line(frame, (0, bottom_pivot),
             (FRAME_WIDTH, bottom_pivot), (255, 255, 255), 2)


def calculate_distance(point1, point2):
    """
    Fungsi untuk menghitung jarak Euclidean antara dua titik.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_mid_point(point1, point2):
    """
    Fungsi untuk menghitung titik tengah antara dua titik.
    """
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)


def main():

    # urgent
    mid_point = 320, 320

    # Load YOLOv8 model
    model = YOLO(MODEL_PATH)

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()  # Ambil frame dari video
            if not ret:
                break

            # Resize frame to 640x640
            resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Catat waktu sebelum deteksi
            start_time = time.time()

            # Menggunakan model.predict() dengan source dari resized_frame dan imgsz 640
            results = model.predict(source=resized_frame,
                                    imgsz=640, conf=0.6, iou=0.7, max_det=8, device='0')

            # Catat waktu setelah deteksi
            end_time = time.time()

            # Hitung waktu yang berlalu per frame dan FPS
            elapsed_time = end_time - start_time
            fps = 1 / elapsed_time

            # Variabel untuk menyimpan objek terdekat per kelas
            closest_green_ball = None
            closest_red_ball = None
            closest_green_distance = float('inf')
            closest_red_distance = float('inf')

            # Ekstraksi manual dari bounding box, confidence scores, dan class labels
            for result in results:
                for box in result.boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    # Hitung titik tengah dari bounding box
                    center_x_box = (x1 + x2) / 2
                    center_y_box = (y1 + y2) / 2
                    center_box = (center_x_box, center_y_box)

                    # Tentukan warna berdasarkan class ID menggunakan class_colors
                    color = class_colors[cls] if cls < len(
                        class_colors) else (255, 255, 255)

                    # Print detected object details
                    detected_class_name = class_names.get(cls, 'Unknown')
                    print(
                        f"Detected object: {detected_class_name}, Confidence: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

                    # Hitung jarak ke pivot point
                    distance_to_pivot = calculate_distance(
                        center_box, pivot_point)

                    # Filter berdasarkan posisi Y (harus di atas bottom_pivot)
                    if center_y_box < bottom_pivot:
                        # Jika class 'green_ball', periksa jarak dan simpan yang terdekat
                        if cls == 0 and distance_to_pivot < closest_green_distance:
                            closest_green_distance = distance_to_pivot
                            closest_green_ball = (
                                x1, y1, x2, y2, center_box, color)

                        # Jika class 'red_ball', periksa jarak dan simpan yang terdekat
                        if cls == 1 and distance_to_pivot < closest_red_distance:
                            closest_red_distance = distance_to_pivot
                            closest_red_ball = (
                                x1, y1, x2, y2, center_box, color)

                    # Gambarkan kotak deteksi dan teks dengan warna yang sesuai
                    cv2.rectangle(resized_frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), color, 2)
                    label = f'{detected_class_name} {conf:.2f}'
                    cv2.putText(resized_frame, label, (int(x1), int(
                        y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Jika ada green_ball dan red_ball terdekat
            if closest_green_ball is not None and closest_red_ball is not None:
                # Dapatkan titik tengah dari kedua bola
                _, _, _, _, green_center, _ = closest_green_ball
                _, _, _, _, red_center, _ = closest_red_ball

                cv2.circle(resized_frame, (int(green_center[0]), int(
                    green_center[1])), 5, (0, 255, 0), -1)  # Titik hijau

                cv2.circle(resized_frame, (int(red_center[0]), int(
                    red_center[1])), 5, (0, 0, 255), -1)  # Titik merah

                # Hitung titik tengah di antara kedua bola
                mid_point = calculate_mid_point(green_center, red_center)

                # Gambar garis yang menghubungkan titik tengah green_ball dan red_ball
                cv2.line(resized_frame, (int(green_center[0]), int(green_center[1])),
                         (int(red_center[0]), int(red_center[1])), (0, 255, 255), 2)

                # Gambar garis dari titik tengah dua bola ke pivot_point
                cv2.line(resized_frame, (int(mid_point[0]), int(mid_point[1])),
                         pivot_point, (0, 255, 255), 2)

                # Gambarkan titik biru di tengah-tengah antara green_ball dan red_ball
                cv2.circle(resized_frame, (int(mid_point[0]), int(mid_point[1])),
                           5, (255, 0, 0), -1)  # Titik biru

            # Gambar garis tengah pada frame
            draw_center_line(resized_frame, fps, closest_green_ball,
                             closest_red_ball, mid_point)

            # Display the frame with the detections
            cv2.imshow('ASV-KKI 2024', resized_frame)

            # lurus
            if mid_point[0] > 250 and mid_point[0] < 390:
                override_rc_channels(connection, rc1_value_awal,
                                     rc2_value_awal, rc3_value_lurus, rc4_value_awal)
            # kiri s
            elif mid_point[0] > 100 and mid_point[0] < 250:
                override_rc_channels(connection, rc1_value_kiri_sitik,
                                     rc2_value_awal, rc3_value_lurus, rc4_value_awal)
            # kiri m
            elif mid_point[0] > 0 and mid_point[0] < 100:
                override_rc_channels(connection, rc1_value_kiri_mentok,
                                     rc2_value_awal, rc3_value_lurus, rc4_value_awal)
            # kanan s
            elif mid_point[0] > 390 and mid_point[0] < 540:
                override_rc_channels(connection, rc1_value_kanan_sitik,
                                     rc2_value_awal, rc3_value_lurus, rc4_value_awal)
            elif mid_point[0] > 540 and mid_point[0] < 640:
                override_rc_channels(connection, rc1_value_kanan_mentok,
                                     rc2_value_awal, rc3_value_lurus, rc4_value_awal)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Menutup koneksi dengan aman
        connection.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
