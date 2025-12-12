import uuid
import sys
import os
import argparse

# Add parent dir for db.py
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from db import get_connection

def schedule_job(video_path):
    print(f"Scheduling job for {video_path}...")
    try:
        cluster, collection = get_connection("video-analytics")
        job_id = f"job::{uuid.uuid4()}"
        job_doc = {
            "type": "video_job",
            "video_path": video_path,
            "status": "pending",
            "created_at": str(uuid.uuid1())
        }
        collection.upsert(job_id, job_doc)
        print(f"✅ Job Scheduled Successfully: {job_id}")
    except Exception as e:
        print(f"❌ Failed to schedule job: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to video file")
    args = parser.parse_args()
    
    schedule_job(args.video_path)
