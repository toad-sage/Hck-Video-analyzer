import uuid
import json
import urllib.request
import urllib.parse
import ssl

class JobScheduler:
    def schedule_job(self, video_path):
        job_id = f"job::{str(uuid.uuid4())}"
        
        # SQL++ Insert Statement
        # We assume `video-analytics` is the dataset/collection name available in Analytics
        # If it's a link, we might need to write to the link or use a different method.
        # But Analytics usually supports INSERT if it's a standalone collection or connected correctly.
        # Wait, Analytics is usually Read-Only for Links. 
        # But we can try hitting the QUERY Service (8093) if available, or Analytics (9600) if it supports storage.
        
        # Let's assume we hit the Analytics endpoint.
        
        statement = (
            f'INSERT INTO `video-analytics` (KEY, VALUE) '
            f'VALUES ("{job_id}", {{ '
            f'  "type": "video_job", '
            f'  "video_path": "{video_path}", '
            f'  "status": "pending", '
            f'  "created_at": "{str(uuid.uuid1())}" '
            f'}})'
        )
        
        # Use localhost:9600 (Analytics)
        url = "http://localhost:9600/analytics/service"
        data = urllib.parse.urlencode({'statement': statement}).encode('utf-8')
        req = urllib.request.Request(url, data=data)
        
        # Basic Auth: couchbase:couchbase
        req.add_header("Authorization", "Basic Y291Y2hiYXNlOmNvdWNoYmFzZQ==")
        
        # Context to ignore SSL if needed
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(req, context=ctx) as response:
                result = response.read().decode('utf-8')
                return f"Job Scheduled: {job_id}"
        except Exception as e:
            # Try to return the response body for debugging
            return f"Error scheduling job: {str(e)}"
