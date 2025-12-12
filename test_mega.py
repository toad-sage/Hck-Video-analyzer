import sys
import os

# Add mega pyz
sys.path.insert(0, 'video-processor-mega.pyz')

try:
    import main_udf
    print("Imported main_udf successfully.")
    
    handler = main_udf.UDFHandler()
    print("Initialized Handler.")
    
    # Path to video
    video_path = "../car-wash-adv.mp4"
    
    print(f"Analyzing Car Wash {video_path}...")
    res = handler.analyze_carwash(video_path)
    print(f"Result: {res}")

except Exception as e:
    print(f"FAILED: {e}")

