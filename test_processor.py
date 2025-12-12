import sys
import os

sys.path.insert(0, 'video-processor-lib.pyz')

try:
    import main_udf
    print("Imported main_udf successfully.")
    
    handler = main_udf.UDFHandler()
    print("Initialized Handler.")
    
    # Path to video (relative to where we run this script)
    video_path = "../mvideo.mp4"
    
    print(f"Processing {video_path}...")
    res = handler.analyze(video_path)
    print(f"Result: {res}")

except Exception as e:
    print(f"FAILED: {e}")


