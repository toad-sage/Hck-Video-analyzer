from ultralytics import YOLO
from video_ai import GenerativeAI
from kv_writer import KVWriter
import cv2
import os
import time

class UDFHandler:
    def __init__(self):
        self.writer = KVWriter()
        self.gen_ai = None # Lazy load

    def _get_gen_ai(self):
        if self.gen_ai is None:
            self.gen_ai = GenerativeAI()
        return self.gen_ai

    def analyze_carwash(self, video_path):
        """
        Specific UDF function to extract Car Wash info.
        """
        prompt = (
            "Analyze this image. Identify any businesses or services. "
            "Output ONLY a valid JSON object with keys: 'shop_type', 'shop_name', 'phonenumber', 'address'. "
            "If a field is not visible, use null. Do not add markdown formatting."
        )
        
        processor = self._get_gen_ai()
        video_name = os.path.basename(video_path)
        count = 0
        
        try:
            for result in processor.process_video_captioning(video_path, prompt=prompt, json_mode=True):
                # Construct Document
                doc = {
                    "source": "udf_carwash_extractor",
                    "video_name": video_name,
                    "frame_id": result['frame_id'],
                    "attributes": result, # Contains 'extracted_data'
                    "timestamp": time.time()
                }
                
                self.writer.write_frame(video_name, result['frame_id'], doc)
                count += 1
                
            return f"Success: Extracted {count} frames from {video_name}"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def analyze_yolo(self, video_path):
        # ... (Keep the previous YOLO logic here if needed, or just use the new one)
        return "YOLO Logic Placeholder"
