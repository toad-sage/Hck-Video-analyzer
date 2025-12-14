import os
import tempfile
from typing import Optional
from uuid import uuid4

import boto3
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from video_ai import GenerativeAI, QWEN_F1_STRATEGY_PROMPT
from kv_writer import KVWriter


app = FastAPI(title="Video Analytics Sidecar API")

# Singletons for reuse across requests
gen_ai = GenerativeAI()
kv_writer = KVWriter()


class AnalyzeRequest(BaseModel):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    region_name: str = "ap-south-1"
    bucket: Optional[str] = None
    key: Optional[str] = None  # S3 object key, e.g. "videos/myvideo.mp4"
    local_path: Optional[str] = None
    json_mode: bool = False
    prompt: Optional[str] = None


def _download_from_s3(req: AnalyzeRequest) -> str:
    """
    Download the requested S3 object to a local temp file and return its path.
    """
    s3 = boto3.client(
        "s3",
        region_name=req.region_name,
        aws_access_key_id=req.aws_access_key_id,
        aws_secret_access_key=req.aws_secret_access_key,
    )

    suffix = os.path.splitext(req.key)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    s3.download_file(req.bucket, req.key, tmp_path)
    return tmp_path


def _extract_audio_transcript(request_id: str, video_path: str) -> Optional[str]:
    """
    Extract audio from the video and run speech-to-text to get a transcript.

    Uses moviepy to dump a WAV file and openai-whisper to transcribe it.
    """
    try:
        from moviepy import VideoFileClip  # type: ignore
        import whisper  # type: ignore
    except ImportError as e:
        print(f"[{request_id}] Audio transcription disabled, missing dependency: {e}")
        return None

    fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    transcript = None
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio is None:
            print(f"[{request_id}] No audio track found in video.")
            clip.close()
            return None

        audio.write_audiofile(audio_path, logger=None)
        clip.close()

        try:
            model = whisper.load_model("small")
            result = model.transcribe(audio_path)
            transcript = (result.get("text") or "").strip() or None
        except Exception as e:
            print(f"[{request_id}] Audio transcription error: {e}")
            transcript = None

    except Exception as e:
        print(f"[{request_id}] Audio extraction/transcription pipeline error: {e}")
        transcript = None
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as cleanup_err:
            print(f"[{request_id}] Audio temp file cleanup error: {cleanup_err}")

    return result


def _process_video_job(
    request_id: str,
    tmp_path: str,
    video_name: str,
    json_mode: bool,
    prompt: Optional[str],
    cleanup_file: bool = True,
):
    """
    Run multimodal analysis (audio + video) on the downloaded file
    and push results to Couchbase KV.
    """
    try:
        effective_prompt = prompt or QWEN_F1_STRATEGY_PROMPT

        # 1) Audio pipeline: extract + transcribe whole video once
        transcript_result = _extract_audio_transcript(request_id, tmp_path)
        print(f"[{request_id}] Transcript result: {transcript_result}")
        full_transcript_text = None
        audio_segments = []

        if transcript_result:
            full_transcript_text = transcript_result.get("text", "").strip()
            audio_segments = transcript_result.get("segments", [])

        if full_transcript_text:
            # Ensure embedder is ready so we can store a vector for the transcript too
            try:
                gen_ai.load_model()
                audio_doc = {
                    "type": "audio_transcript",
                    "video_name": video_name,
                    "request_id": request_id,
                    "transcript": full_transcript_text,
                    "vector": gen_ai.embedder.encode(full_transcript_text).tolist()
                    if gen_ai.embedder
                    else None,
                }
            except Exception as e:
                print(f"[{request_id}] Failed to embed transcript, storing text only: {e}")
                audio_doc = {
                    "type": "audio_transcript",
                    "video_name": video_name,
                    "request_id": request_id,
                    "transcript": full_transcript_text,
                }

            # Use a synthetic 'frame_id' key suffix for transcript
            kv_writer.write_frame(video_name, "audio", audio_doc)

        # 2) Frame-wise visual + text pipeline (Qwen2-VL)
        for res in gen_ai.process_video_captioning(
            tmp_path,
            prompt=effective_prompt,
            json_mode=json_mode,
            audio_segments=audio_segments
        ):
            # Defensive checks
            if not isinstance(res, dict):
                print(f"[{request_id}] Skipping non-dict result: {type(res)}")
                continue
            if res.get("error"):
                print(f"[{request_id}] Qwen error frame: {res['error']}")
                continue

            frame_id = res.get("frame_id")
            if frame_id is None:
                print(f"[{request_id}] Skipping result without frame_id: {res}")
                continue

            # Enrich and write to KV
            # We flatten the extracted fields so they are directly queryable,
            # and still keep raw_text for debugging if needed.
            if json_mode and res.get("extracted_data"):
                # JSON-analysis mode: promote extracted_data fields
                doc = {
                    "type": "frame_analysis",
                    "video_name": video_name,
                    "frame_id": frame_id,
                    "request_id": request_id,
                }
                doc.update(res["extracted_data"])
                # Attach vector if present
                if "vector" in res:
                    doc["vector"] = res["vector"]
                # Optional: keep raw_text for debugging
                doc["raw_text"] = res.get("raw_text")
                doc["timestamp"] = res.get("timestamp")
                doc["audio_context"] = res.get("audio_context")
            else:
                # Caption mode or JSON parse failure: keep payload structure
                doc = {
                    "type": "frame_caption",
                    "video_name": video_name,
                    "frame_id": frame_id,
                    "request_id": request_id,
                    "payload": res,
                }

            ok = kv_writer.write_frame(video_name, frame_id, doc)
            if not ok:
                print(f"[{request_id}] Failed to write frame {frame_id} for {video_name}")

    except Exception as e:
        print(f"[{request_id}] Video processing error for {video_name}: {e}")
    finally:
        # Clean up local file if requested
        if cleanup_file:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception as cleanup_err:
                print(f"[{request_id}] Temp file cleanup error: {cleanup_err}")


@app.post("/analyze")
def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Kick off analysis of a video stored in S3 or locally.
    """
    # Determine source
    if request.local_path:
        if not os.path.exists(request.local_path):
            raise HTTPException(status_code=400, detail=f"Local file not found: {request.local_path}")
        tmp_path = request.local_path
        video_name = os.path.basename(tmp_path)
        cleanup_file = False
        s3_bucket = "local"
        s3_key = request.local_path
    else:
        # S3 Mode - Validate S3 params
        if not (request.aws_access_key_id and request.aws_secret_access_key and request.bucket and request.key):
            raise HTTPException(
                status_code=400, 
                detail="Missing S3 credentials. Provide 'local_path' OR ('aws_access_key_id', 'aws_secret_access_key', 'bucket', 'key')."
            )
        # Download immediately so the HTTP request can return once job is scheduled
        tmp_path = _download_from_s3(request)
        video_name = os.path.basename(request.key)
        cleanup_file = True
        s3_bucket = request.bucket
        s3_key = request.key

    # Generate a request/job id for tracking
    request_id = str(uuid4())

    background_tasks.add_task(
        _process_video_job,
        request_id=request_id,
        tmp_path=tmp_path,
        video_name=video_name,
        json_mode=request.json_mode,
        prompt=request.prompt,
        cleanup_file=cleanup_file,
    )

    return {
        "status": "ok",
        "requestId": request_id,
        "video_name": video_name,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
    }


# Optional root for quick health check
@app.get("/")
def health():
    return {"status": "ok"}


