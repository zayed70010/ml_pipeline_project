import shutil
import os

# تعريف المسارات
source_path = "../data/vehicles_dataset_prepared.csv"
destination_path = "vehicles_dataset_prepared.csv"

# نسخ الملف من مجلد data إلى lab3
try:
    shutil.copy(source_path, destination_path)
    print(f"✅ File copied to: {destination_path}")
except FileNotFoundError:
    print("❌ Source file not found. Make sure it exists at: data/vehicles_dataset_prepared.csv")
except Exception as e:
    print(f"❌ Error during file copy: {e}")
