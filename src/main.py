from experiments.bank import main as bank_main
from experiments.face_temp import main as face_temp_main
from experiments.parkinsons import main as parkinsons_main
from experiments.thyroid_cancer import main as thyroid_cancer_main
from experiments.wine import main as wine_main


def main():
    print("\n" + "=" * 80)
    print("RUNNING BANK MARKETING EXPERIMENTS")
    print("=" * 80)
    bank_main()

    print("\n" + "=" * 80)
    print("RUNNING FACE TEMPERATURE EXPERIMENTS")
    print("=" * 80)
    face_temp_main()

    # print("\n" + "=" * 80)
    # print("RUNNING PARKINSON'S TELEMONITORING EXPERIMENTS")
    # print("=" * 80)
    # parkinsons_main()

    # print("\n" + "=" * 80)
    # print("RUNNING FACE TEMPERATURE EXPERIMENTS")
    # print("=" * 80)
    # face_temp_main()

    # print("\n" + "=" * 80)
    # print("RUNNING THYROID CANCER EXPERIMENTS")
    # print("=" * 80)
    # thyroid_cancer_main()

    # print("\n" + "=" * 80)
    # print("RUNNING WINE QUALITY EXPERIMENTS")
    # print("=" * 80)
    # wine_main()


if __name__ == "__main__":
    main()