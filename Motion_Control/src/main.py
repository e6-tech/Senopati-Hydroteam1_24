from detection import BallDetector


def main():
    # Inisialisasi detektor bola
    detector = BallDetector(
        model_path='../../models/best.pt',
        video_path='../../B.mp4',
        max_lr=100
    )

    try:
        print("trues")
        # Panggil fungsi `detect` dari `BallDetector`
        detector.detect(set_device='cpu')
        # data = detector.navigation_data()
        # print(f"Posisi X: {data}")
    except KeyboardInterrupt:
        # Jika user menekan Ctrl + C, program berhenti
        print("Deteksi dihentikan.")
    except Exception as e:
        # Tangkap dan cetak kesalahan lain
        print(f"Terjadi kesalahan: {e}")


if __name__ == "__main__":
    main()
