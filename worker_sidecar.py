import time
import json
import traceback
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.options import QueryOptions
from datetime import timedelta

# Import your heavy AI logic
import sys
import os

# Add parent directory to path to find processor.py and db.py
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Add current directory
sys.path.append(os.getcwd())

from video_ai import GenerativeAI

# Config
CB_HOST = "couchbase://127.0.0.1:10182"
CB_USER = "Administrator"
CB_PASS = "password"
BUCKET_NAME = "video-analytics"

def get_couchbase_connection():
    cluster = Cluster(CB_HOST, ClusterOptions(PasswordAuthenticator(CB_USER, CB_PASS)))
    bucket = cluster.bucket(BUCKET_NAME)
    collection = bucket.default_collection()
    return cluster, bucket, collection

def process_jobs():
    cluster, bucket, collection = get_couchbase_connection()
    
    # Initialize AI Models once
    print("Initializing AI Models...")
    gen_ai = GenerativeAI()
    print("Models Ready. Waiting for jobs...")

    while True:
        try:
            # Poll for pending jobs
            # Using N1QL for simplicity. In prod, use DCP or Eventing.
            query = "SELECT META().id, * FROM `video-analytics` WHERE type='video_job' AND status='pending' LIMIT 1"
            rows = cluster.query(query, QueryOptions(metrics=True))
            
            job = None
            for row in rows:
                job = row
                break
            
            if job:
                doc_id = job['id']
                data = job['video-analytics']
                video_path = data.get('video_path')
                
                print(f"Found Job: {doc_id} for video: {video_path}")
                
                # Update status to processing
                collection.upsert(doc_id, {**data, "status": "processing"})
                
                try:
                    # Run AI - Qwen only (caption + vector embedding)
                    print(f"Running Qwen on {video_path}...")
                    
                    # Qwen Captioning: gen_ai.process_video_captioning yields frame results
                    summary = ""
                    for res in gen_ai.process_video_captioning(video_path):
                        # Defensive checks: only handle well-formed dicts with frame_id
                        if not isinstance(res, dict):
                            print(f"Skipping unexpected Qwen result type: {type(res)}")
                            continue
                        if res.get("error"):
                            print(f"Qwen frame error: {res['error']}")
                            continue
                        if "frame_id" not in res:
                            print(f"Skipping Qwen result without frame_id: {res}")
                            continue

                        # Store frame result immediately
                        key = f"{os.path.basename(video_path)}::{res['frame_id']}::caption"
                        # Add metadata
                        res["type"] = "frame_caption"
                        res["video_name"] = os.path.basename(video_path)
                        res["processed_at"] = str(time.time())

                        try:
                            collection.upsert(key, res)
                            print(f"Stored caption for frame {res['frame_id']}")
                        except Exception as e:
                            print(f"Error storing caption: {e}")

                        summary = str(res)  # Keep last for summary
                    
                    print("AI Processing Complete.")
                    
                    # Update Job Status
                    collection.upsert(doc_id, {
                        **data, 
                        "status": "completed", 
                        "result": summary
                    })
                    print(f"Job {doc_id} Completed.")
                    
                except Exception as e:
                    print(f"AI Failed: {e}")
                    traceback.print_exc()
                    collection.upsert(doc_id, {**data, "status": "failed", "error": str(e)})
            
            else:
                # No jobs, sleep
                time.sleep(2)
                
        except Exception as e:
            print(f"Worker Loop Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    process_jobs()

