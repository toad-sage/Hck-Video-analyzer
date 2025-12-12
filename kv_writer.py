from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from datetime import timedelta
import time
import os

class KVWriter:
    def __init__(self):
        # Use the known working configuration
        self.host = "couchbase://127.0.0.1:10182"
        self.user = "Administrator"
        self.password = "password"
        self.bucket_name = "video-analytics"
        self._cluster = None
        self._collection = None

    def connect(self):
        if self._cluster:
            return
        try:
            auth = PasswordAuthenticator(self.user, self.password)
            opts = ClusterOptions(auth)
            opts.apply_profile('wan_development')
            
            self._cluster = Cluster(self.host, opts)
            self._cluster.wait_until_ready(timedelta(seconds=5))
            
            bucket = self._cluster.bucket(self.bucket_name)
            self._collection = bucket.default_collection()
        except Exception as e:
            print(f"KV Connect Error: {e}")
            raise

    def write_frame(self, video_name, frame_id, data):
        self.connect()
        key = f"{video_name}::{frame_id}"
        try:
            # Upsert (Create or Update)
            self._collection.upsert(key, data)
            return True
        except Exception as e:
            print(f"KV Write Error: {e}")
            return False


