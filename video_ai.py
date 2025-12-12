import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
import cv2
import os
from PIL import Image
import json
import re

QWEN_F1_STRATEGY_PROMPT = (
    "You are an expert Formula 1 race strategist and data analyst.\n\n"
    "Users will ask questions like:\n"
    "- \"give me the timestamp when Norris went for tyre change in the pit\"\n"
    "- \"how many times was McLaren ahead of Ferrari?\"\n"
    "- \"who finished on the podium and how did their strategies differ?\"\n\n"
    "From ONLY this ONE frame of the race video, infer as much as you reasonably can about the race "
    "situation, strategy, pit lane activity and podium context, and output a single JSON object "
    "with the following structure:\n\n"
    "{\n"
    "  \"frame_summary\": {\n"
    "    \"high_level_description\": \"<short natural language description of what is happening in this frame>\",\n"
    "    \"session_phase\": \"<one of: formation_lap | race_start | early_race | mid_race | late_race | safety_car | pit_stop_sequence | podium_ceremony | unknown>\",\n"
    "    \"track_name_or_location\": \"<best guess or null>\",\n"
    "    \"lap_hint\": \"<visible lap number if shown on HUD/graphics, else null>\",\n"
    "    \"video_timestamp_hint\": \"<any visible race time / lap timer / session clock / overlay that indicates WHEN in the race this frame occurs, e.g. 'Lap 27/58 at 01:12:34 race time'; if nothing visible, null>\"\n"
    "  },\n\n"
    "  \"visible_cars\": [\n"
    "    {\n"
    "      \"car_index_in_frame\": <0-based index of this car in this frame>,\n"
    "      \"team_name\": \"<best guess team name: McLaren, Ferrari, Red Bull, Mercedes, etc., or null>\",\n"
    "      \"driver_name\": \"<best guess: Lando Norris, Max Verstappen, Lewis Hamilton, etc., or null>\",\n"
    "      \"car_number\": \"<visible car number if readable, else null>\",\n"
    "      \"livery_description\": \"<short description of colours/logos to help identify the car>\",\n"
    "      \"current_position_hint\": \"<if you can see on-screen graphics / car numbering / podium context, guess race position like P1, P2, P3; else null>\",\n"
    "      \"tyre_compound_hint\": \"<best guess: soft | medium | hard | intermediate | wet | unknown>\",\n"
    "      \"on_pit_lane\": <true | false>,\n"
    "      \"is_pitting_in_this_frame\": <true | false>,\n"
    "      \"pit_crew_activity_hint\": \"<describe if wheels are off, jacks, lollipop, pit lights, or mechanics around the car; else null>\"\n"
    "    }\n"
    "  ],\n\n"
    "  \"strategy_and_race_context\": {\n"
    "    \"overtake_or_defence_hint\": \"<if you see a clear attack/defence or overtake, describe it; else null>\",\n"
    "    \"safety_car_or_flags_hint\": \"<yellow_flag | safety_car | virtual_safety_car | red_flag | green_flag | none | unknown>\",\n"
    "    \"team_strategy_hint\": \"<any visible evidence of undercut/overcut/tyre change/long stint or stacked pit stops; if not obvious, null>\",\n"
    "    \"key_strategy_events_visible\": [\n"
    "      \"<short bullet-like descriptions of events suggested by THIS frame, e.g. 'Norris enters pit lane for tyre change', 'Ferrari double-stack pit stop', 'Podium ceremony with Norris P1, Verstappen P2, Hamilton P3'; or empty array if none>\"\n"
    "    ]\n"
    "  },\n\n"
    "  \"podium_and_points_hint\": {\n"
    "    \"is_podium_scene\": <true | false>,\n"
    "    \"probable_podium_order\": [\n"
    "      {\n"
    "        \"position\": 1,\n"
    "        \"driver_name\": \"<if clearly shown or strongly implied, else null>\",\n"
    "        \"team_name\": \"<if clearly shown or implied, else null>\"\n"
    "      },\n"
    "      {\n"
    "        \"position\": 2,\n"
    "        \"driver_name\": \"<or null>\",\n"
    "        \"team_name\": \"<or null>\"\n"
    "      },\n"
    "      {\n"
    "        \"position\": 3,\n"
    "        \"driver_name\": \"<or null>\",\n"
    "        \"team_name\": \"<or null>\"\n"
    "      }\n"
    "    ],\n"
    "    \"championship_points_hint\": \"<if on-screen graphics show points gained or standings, write a brief summary; else null>\"\n"
    "  },\n\n"
    "  \"text_on_screen\": {\n"
    "    \"raw_hud_text\": \"<any timing/position/tyre/DRS/sector or lap counter data you can read or approximate>\",\n"
    "    \"sponsor_or_series_logos\": [\n"
    "      \"<e.g. FIA, F1, Pirelli, etc.>\"\n"
    "    ]\n"
    "  },\n\n"
    "  \"confidence\": {\n"
    "    \"driver_id_confidence\": \"<low | medium | high>\",\n"
    "    \"team_id_confidence\": \"<low | medium | high>\",\n"
    "    \"strategy_inference_confidence\": \"<low | medium | high>\",\n"
    "    \"podium_inference_confidence\": \"<low | medium | high>\"\n"
    "  }\n"
    "}\n\n"
    "STRICT RULES:\n"
    "- Output ONLY valid JSON, no markdown, no comments, no extra text.\n"
    "- If something is not visible or cannot be inferred from THIS ONE IMAGE, set it to null or an empty array. "
    "Do NOT hallucinate precise details like exact lap numbers or timestamps that are not visible.\n"
    "- Prefer conservative guesses with low/medium confidence rather than making up unsupported facts.\n"
    "- For driver_name and team_name:\n"
    "  - Only set driver_name if you can clearly read a driver name, car number, or very distinctive helmet + car "
    "and are reasonably confident. Otherwise set driver_name to null and driver_id_confidence to \"low\".\n"
    "  - Only set team_name if you can see a clear team logo, livery text, or an on-screen team label. "
    "If unsure, set team_name to null and team_id_confidence to \"low\".\n"
    "  - Never output obviously impossible combinations (for example: 'Max Verstappen' driving for 'Ferrari'). "
    "If you are unsure, prefer null over guessing.\n"
    "- For lap_hint and video_timestamp_hint:\n"
    "  - Only use values that are directly visible in on-screen HUD/graphics. "
    "Do NOT invent exact lap numbers or timestamps if you cannot read them.\n"
    "- For key_strategy_events_visible:\n"
    "  - Only describe strategy events that are clearly suggested by THIS frame (or its immediate overlays), "
    "such as a visible pit stop, podium ceremony, or safety car on track. Do NOT describe the whole race history.\n"
)

def _parse_qwen_json(output_text: str):
    """
    Best-effort parser for Qwen JSON outputs.

    - Strips ```json fences
    - Extracts the substring between the first '{' and last '}'
    - Attempts a strict json.loads
    - On failure, tries to fix common issues like trailing commas
    """
    # Remove markdown fences if still present
    clean_text = re.sub(r'```json|```', '', output_text).strip()

    # Keep only the innermost JSON object if there is extra commentary
    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        clean_text = clean_text[start : end + 1]

    # First attempt: raw
    try:
        return json.loads(clean_text)
    except Exception:
        # Second attempt: remove trailing commas before } or ]
        fixed = re.sub(r",(\s*[}\]])", r"\1", clean_text)
        return json.loads(fixed)


class GenerativeAI:
    def __init__(self):
        self.model = None
        self.processor = None
        self.embedder = None
        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"

    def load_model(self):
        if self.model:
            return
        
        print(f"Loading Qwen2-VL on {self.device}...")
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "mps" else None
            )
            if self.device == "mps":
                self.model = self.model.to(self.device)
                
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            
            print("Loading CLIP...")
            self.embedder = SentenceTransformer('clip-ViT-B-32', device=self.device)
            
        except Exception as e:
            print(f"GenAI Load Error: {e}")
            raise

    def process_video_captioning(self, video_path, prompt=QWEN_F1_STRATEGY_PROMPT, json_mode=False):
        if not os.path.exists(video_path):
            yield {"error": f"File {video_path} not found"}
            return

        self.load_model()
        
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        
        # Adaptive sampling state
        prev_frame_gray = None
        last_processed_id = -1
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Adaptive Sampling Logic
            should_process = False
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame_gray is None:
                should_process = True
            else:
                diff = cv2.mean(cv2.absdiff(current_frame_gray, prev_frame_gray))[0]
                if diff > 15.0 or (frame_id - last_processed_id) > 150:
                    should_process = True
            
            if not should_process:
                frame_id += 1
                continue

            # Update state
            prev_frame_gray = current_frame_gray
            last_processed_id = frame_id

            # Inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Allow more room so long JSON responses don't get cut too early
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Result Construction
            result = {
                "frame_id": frame_id,
                "raw_text": output_text
            }

            if json_mode:
                try:
                    json_data = _parse_qwen_json(output_text)
                    # Filter empty/null results
                    if any(json_data.values()):
                        result["extracted_data"] = json_data
                        # Also compute an embedding over the structured content so it is searchable
                        try:
                            if self.embedder:
                                embed_text = json.dumps(json_data, ensure_ascii=False)
                                result["vector"] = self.embedder.encode(embed_text).tolist()
                        except Exception as e:
                            # Don't fail the frame just because embedding failed
                            result["vector_error"] = str(e)
                    else:
                        result = None  # Skip empty
                except Exception:
                    # JSON invalid (often because generation was cut off) – still embed raw_text
                    result["error_parse"] = True
                    if self.embedder:
                        try:
                            result["vector"] = self.embedder.encode(output_text).tolist()
                        except Exception as e:
                            result["vector_error"] = str(e)
            else:
                result["caption"] = output_text
                result["vector"] = self.embedder.encode(output_text).tolist()

            if result:
                yield result

            frame_id += 1

        cap.release()

