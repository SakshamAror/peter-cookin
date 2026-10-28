"""Local camera food-labeling prototype. Press q or Esc to quit."""

from __future__ import annotations

import time
import base64
import io
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from dotenv import load_dotenv
from PIL import Image, ImageTk

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

_ELEVENLABS_CLIENT = None


def speak_response(text: str) -> None:
    """Synthesize and play one Gemini response with the configured ElevenLabs voice."""
    global _ELEVENLABS_CLIENT
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID")
    if not api_key or not voice_id:
        raise RuntimeError("Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env or the environment.")

    from elevenlabs.client import ElevenLabs

    if _ELEVENLABS_CLIENT is None:
        _ELEVENLABS_CLIENT = ElevenLabs(api_key=api_key)
    audio_chunks = _ELEVENLABS_CLIENT.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    audio_bytes = b"".join(audio_chunks)
    if not audio_bytes:
        raise RuntimeError("ElevenLabs returned an empty audio response.")
    print(f"[TTS] ElevenLabs returned {len(audio_bytes):,} bytes for the configured voice.", flush=True)

    # On macOS, afplay is the native speaker player. Keep Pygame as a fallback
    # for environments where the native player is unavailable or fails.
    afplay = shutil.which("afplay")
    if afplay:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3") as audio_file:
                audio_file.write(audio_bytes)
                audio_file.flush()
                print("[TTS] Playing through macOS audio output (afplay)…", flush=True)
                subprocess.run([afplay, audio_file.name], check=True)
            return
        except (OSError, subprocess.SubprocessError):
            print("[TTS] afplay failed; trying Pygame playback…", flush=True)

    import pygame

    if pygame.mixer.get_init() is None:
        pygame.mixer.init(frequency=44100)
    audio_stream = io.BytesIO(audio_bytes)
    print("[TTS] Playing through Pygame…", flush=True)
    pygame.mixer.music.load(audio_stream, "mp3")
    pygame.mixer.music.play()
    playback_clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        playback_clock.tick(20)
    pygame.mixer.music.unload()


MODEL_NAME = "yoloe-11s-seg.pt"  # open-vocabulary detector/segmenter; downloaded on first launch
WINDOW_NAME = "Peter Cookin — Local Camera"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
INFERENCE_SIZE = 960
DETECTOR_CONFIDENCE = 0.005
DISPLAY_CONFIDENCE = 0.005
LEAFY_DISPLAY_CONFIDENCE = 0.005
MILK_DISPLAY_CONFIDENCE = 0.40
LABEL_REVIEW_INTERVAL = 5.0
MAX_GEMINI_CANDIDATES = 10
MAX_FOOD_AUDIT_CANDIDATES = 8
TRACKER_CONFIG = Path(__file__).with_name("food_bytetrack.yaml")
FOOD_CLASSES = [
    "apple", "banana", "orange", "lemon", "lime", "pear", "strawberry",
    "blueberry", "grapes", "avocado", "tomato", "potato", "sweet potato",
    "onion", "garlic", "bell pepper", "cucumber", "zucchini", "eggplant",
    "mushroom", "lettuce", "leaf lettuce", "romaine lettuce", "lettuce leaves",
    "salad greens", "spinach", "cabbage", "kale", "broccoli", "carrot",
    "corn", "peas", "green beans", "bread", "bagel", "croissant", "sandwich", "cheese",
    "egg", "milk carton", "gallon milk jug", "coffee creamer", "coffee creamer bottle",
    "coffee mate creamer", "yogurt cup",
    "butter", "stick of butter", "chicken breast", "steak",
    "fish fillet", "shrimp", "bacon", "sausage", "pizza", "pasta",
    "bowl of rice", "donut", "cake", "cereal box", "food can",
    "shredded cheese", "cheese package", "tortillas", "tortilla", "flour tortillas",
    "corn tortillas", "salt", "salt shaker", "pepper", "pepper shaker",
    "black pepper", "ground black pepper",
    "hot sauce", "tomato sauce", "pasta sauce", "marinara sauce", "ketchup",
    "mustard", "mayonnaise", "soy sauce", "barbecue sauce", "salsa",
    "salad dressing", "ranch dressing", "honey", "maple syrup", "jam",
    "peanut butter", "cooking oil", "olive oil", "vegetable oil", "vinegar",
    "spice jar", "seasoning bottle", "seasoning packet", "garlic powder",
    "onion powder", "paprika", "chili powder", "cumin", "cinnamon",
    "oregano", "dried herbs", "spices", "flour", "sugar", "brown sugar",
    "baking powder", "baking soda", "cornstarch", "rice bag", "dry rice",
    "oats", "oatmeal", "beans", "canned beans", "chickpeas", "lentils",
    "canned tomatoes", "tomato paste", "broth", "stock carton", "noodles",
    "ramen", "cooking cream", "sour cream", "cream cheese", "parmesan",
    "mozzarella", "cheddar cheese", "deli meat", "ham", "tofu", "hummus",
    "frozen vegetables", "frozen food package", "food jar", "condiment bottle",
]
OBJECT_PROPOSAL_CLASSES = [
    "object", "food package", "package", "bottle", "carton", "jug", "jar",
    "bag", "box", "container", "can", "shaker", "plastic container",
]
MODEL_CLASSES = FOOD_CLASSES + OBJECT_PROPOSAL_CLASSES
OBJECT_PROPOSAL_CLASS_SET = {name.casefold() for name in OBJECT_PROPOSAL_CLASSES}
SWIFT_OCR = Path(__file__).with_name("read_english_label.swift")
LOW_CONFIDENCE_LEAFY_FOODS = {"lettuce", "spinach", "kale", "cabbage"}
FOOD_CLASS_ALIASES = {
    "leaf lettuce": "lettuce",
    "romaine lettuce": "lettuce",
    "lettuce leaves": "lettuce",
    "salad greens": "lettuce",
    "milk carton": "milk",
    "gallon milk jug": "milk",
    "coffee creamer bottle": "coffee creamer",
    "coffee mate creamer": "coffee creamer",
    "stick of butter": "butter",
    "shredded cheese": "cheese",
    "cheese package": "cheese",
    "tortilla": "tortillas",
    "flour tortillas": "tortillas",
    "corn tortillas": "tortillas",
    "salt shaker": "salt",
    "pepper shaker": "pepper",
    "black pepper": "pepper",
    "ground black pepper": "pepper",
    "spice jar": "seasoning",
    "seasoning bottle": "seasoning",
    "seasoning packet": "seasoning",
    "garlic powder": "seasoning",
    "onion powder": "seasoning",
    "dried herbs": "seasoning",
    "spices": "seasoning",
    "rice bag": "rice",
    "dry rice": "rice",
    "canned beans": "beans",
    "stock carton": "broth",
    "frozen food package": "frozen food",
}


class LocalEnglishOCR:
    """Read package text locally with Apple's Vision OCR in a background process."""

    def __init__(self) -> None:
        self.process = None
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.text = ""
        self.last_submit = 0.0
        try:
            self.process = subprocess.Popen(
                ["swift", str(SWIFT_OCR)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except (FileNotFoundError, OSError):
            print("English label reading unavailable (Swift is not installed). Food detection still works.")
            return
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, frame: np.ndarray) -> None:
        """Queue a small camera image for OCR at most once every 1.5 seconds."""
        if not self.process or self.process.poll() is not None:
            return
        now = time.monotonic()
        if now - self.last_submit < 1.5:
            return
        self.last_submit = now
        height, width = frame.shape[:2]
        if width > 1100:
            frame = cv2.resize(frame, (1100, int(height * 1100 / width)))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return
        try:
            if self.frames.full():
                self.frames.get_nowait()
            self.frames.put_nowait(encoded.tobytes())
        except queue.Empty:
            pass
        except queue.Full:
            pass

    def _worker(self) -> None:
        while self.process and self.process.poll() is None:
            try:
                image_data = self.frames.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                assert self.process.stdin is not None and self.process.stdout is not None
                self.process.stdin.write(base64.b64encode(image_data).decode("ascii") + "\n")
                self.process.stdin.flush()
                result = json.loads(self.process.stdout.readline() or "[]")
                text = " · ".join(str(line).strip() for line in result if str(line).strip())
                with self.lock:
                    self.text = text
            except (BrokenPipeError, OSError, json.JSONDecodeError):
                return

    def latest_text(self) -> str:
        with self.lock:
            return self.text

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def food_name_from_text(text: str) -> str | None:
    """Return an exact food name found in readable package text, if any."""
    normalized = re.sub(r"\s+", " ", text.lower())
    for name in sorted(FOOD_CLASSES, key=len, reverse=True):
        pattern = r"\b" + r"\s+".join(re.escape(word) for word in name.split()) + r"\b"
        if re.search(pattern, normalized):
            return FOOD_CLASS_ALIASES.get(name, name)
    return None


def draw_label(frame: np.ndarray, box: tuple[int, int, int, int], text: str) -> None:
    """Draw a readable label and lightweight perspective wireframe around an item."""
    x1, y1, x2, y2 = box
    front_color = (116, 211, 238)  # warm butter, in BGR
    rear_color = (91, 158, 190)  # muted amber for the receding edges
    shadow_color = (25, 39, 30)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    depth = max(5, min(12, round(min(width, height) * 0.045)))
    front = ((x1, y1), (x2, y2))
    rear = ((x1 + depth, y1 - depth), (x2 + depth, y2 - depth))

    # Offset rear frame and corner joins suggest a shallow 3D box while staying
    # anchored to the detector's live 2D box.
    cv2.rectangle(frame, rear[0], rear[1], shadow_color, 3, cv2.LINE_AA)
    cv2.rectangle(frame, rear[0], rear[1], rear_color, 1, cv2.LINE_AA)
    for point_front, point_rear in (
        ((x1, y1), (x1 + depth, y1 - depth)),
        ((x2, y1), (x2 + depth, y1 - depth)),
        ((x1, y2), (x1 + depth, y2 - depth)),
        ((x2, y2), (x2 + depth, y2 - depth)),
    ):
        cv2.line(frame, point_front, point_rear, shadow_color, 4, cv2.LINE_AA)
        cv2.line(frame, point_front, point_rear, rear_color, 1, cv2.LINE_AA)
    cv2.rectangle(frame, front[0], front[1], shadow_color, 4, cv2.LINE_AA)
    cv2.rectangle(frame, front[0], front[1], front_color, 2, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y1 - text_height - baseline - 10)
    cv2.rectangle(frame, (x1, top), (x1 + text_width + 14, top + text_height + baseline + 8), front_color, -1)
    cv2.putText(frame, text, (x1 + 7, top + text_height + 2), font, scale,
                (24, 34, 28), thickness, cv2.LINE_AA)


class PantryApp:
    """Native local UI for the camera, live food labels, and food request input."""

    def __init__(self, model: YOLO) -> None:
        self.model = model
        self.ocr = LocalEnglishOCR()
        self.stop_event = threading.Event()
        self.frames: queue.Queue[dict] = queue.Queue(maxsize=1)
        self.inference_frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.preview_lock = threading.Lock()
        self.preview_frame: np.ndarray | None = None
        self.preview_sequence = 0
        self.last_rendered_sequence = -1
        self.capture_fps = 0.0
        self.display_fps = 0.0
        self.last_displayed_frame_at = 0.0
        self.gemini_results: queue.Queue[tuple[str, str]] = queue.Queue()
        self.speech_queue: queue.Queue[str] = queue.Queue(maxsize=8)
        self.audit_results: queue.Queue[tuple[dict[int, dict], str, str]] = queue.Queue()
        self.latest: dict | None = None
        self.gemini_busy = False
        self.gemini_client = None
        self.gemini_client_lock = threading.Lock()
        self.initial_audit_started = False
        self.label_audit_in_flight = False
        self.last_label_audit_at = time.monotonic()
        self.reviewed_track_ids: set[int] = set()
        self.confirmed_candidate_track_ids: set[int] = set()
        self.track_label_overrides: dict[int, str] = {}
        self.class_label_overrides: dict[str, str] = {}
        self.ignored_track_ids: set[int] = set()
        self.ignored_class_names: set[str] = set()
        self.last_recipe_context = ""
        self.last_operation_kind = ""
        self.pending_ai_question = ""
        self.status_note = ""
        self.status_note_until = 0.0
        self.image_ref = None
        self.root = tk.Tk()
        self.root.title("Peter Cookin · Pantry Lens")
        self.root.geometry("1600x900")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#18261d")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker = threading.Thread(target=self._camera_loop, daemon=True)
        self.inference_worker = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker.start()
        self.inference_worker.start()
        self.root.after(1, self._refresh)
        self.root.after(2_000, self._run_initial_label_audit)
        self.tts_enabled = bool(
            os.environ.get("ELEVENLABS_API_KEY") and os.environ.get("ELEVENLABS_VOICE_ID")
        )
        if self.tts_enabled:
            self.speech_worker = threading.Thread(target=self._speech_loop, daemon=True)
            self.speech_worker.start()
            print("[TTS] ElevenLabs speech is enabled with the voice ID from .env.", flush=True)
        else:
            self.speech_worker = None
            print("[TTS] Speech disabled; set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID to enable it.", flush=True)

    def _build_ui(self) -> None:
        main_area = tk.Frame(self.root, bg="#18261d")
        main_area.pack(fill="both", expand=True)

        camera_column = tk.Frame(main_area, bg="#18261d")
        camera_column.pack(side="left", fill="both", expand=True)

        self.sidebar = tk.Frame(main_area, bg="#f5f0df", width=270,
                                highlightthickness=1, highlightbackground="#d8ddcf")
        self.sidebar.pack(side="right", fill="y")
        self.sidebar.pack_propagate(False)
        tk.Label(self.sidebar, text="PETER COOKIN", bg="#f5f0df", fg="#275f45",
                 font=("Helvetica", 15, "bold"), padx=16, pady=0).pack(anchor="w", pady=(18, 3))
        tk.Label(self.sidebar, text="YOUR COOKING COMPANION", bg="#f5f0df", fg="#778478",
                 font=("Helvetica", 8, "bold"), padx=17, pady=0).pack(anchor="w", pady=(0, 13))
        tk.Frame(self.sidebar, bg="#d8ddcf", height=1).pack(fill="x", padx=14)
        chat_area = tk.Frame(self.sidebar, bg="#f5f0df")
        chat_area.pack(fill="both", expand=True, padx=(12, 8), pady=10)
        chat_scrollbar = tk.Scrollbar(chat_area, relief="flat", bd=0)
        chat_scrollbar.pack(side="right", fill="y")
        self.ai_chat = tk.Text(
            chat_area, wrap="word", state="disabled", relief="flat", bd=0,
            bg="#f5f0df", fg="#34463a", font=("Helvetica", 11),
            padx=5, pady=5, spacing1=3, spacing3=7,
            yscrollcommand=chat_scrollbar.set,
        )
        self.ai_chat.pack(side="left", fill="both", expand=True)
        chat_scrollbar.configure(command=self.ai_chat.yview)
        self.ai_chat.tag_configure("you", foreground="#8b5f2b", font=("Helvetica", 9, "bold"))
        self.ai_chat.tag_configure("peter", foreground="#275f45", font=("Helvetica", 9, "bold"))
        self.ai_chat.tag_configure("message", foreground="#34463a", font=("Helvetica", 11), spacing3=12)
        self.ai_chat.tag_configure("error", foreground="#a33e32", font=("Helvetica", 10))
        self.ai_chat.configure(state="normal")
        self.ai_chat.insert("end", "Ask for a recipe or cooking progress check.\nThe answer will appear here.", "message")
        self.ai_chat.configure(state="disabled")

        camera_area = tk.Frame(camera_column, bg="#18261d")
        camera_area.pack(fill="both", expand=True)
        self.camera_view = tk.Label(camera_area, text="Starting camera and detector…",
                                    bg="#18261d", fg="#f4f0df", font=("Helvetica", 14),
                                    bd=0, highlightthickness=0)
        self.camera_view.pack(fill="both", expand=True)
        self.brand = tk.Label(camera_area, text="✳  PETER COOKIN", bg="#20372a",
                              fg="#f4f0df", font=("Helvetica", 10, "bold"),
                              padx=12, pady=8, bd=0)
        self.brand.place(relx=0.015, y=14, anchor="nw")
        self.status = tk.Label(camera_area, text="Starting camera…", bg="#20372a",
                               fg="#dce8d9", font=("Helvetica", 10, "bold"),
                               padx=12, pady=8, bd=0)
        self.status.place(relx=0.985, y=14, anchor="ne")

        prompt_bar = tk.Frame(camera_column, bg="#18261d", padx=18, pady=12)
        prompt_bar.pack(fill="x", side="bottom")
        self.request_text = tk.Entry(prompt_bar, font=("Helvetica", 14), relief="flat",
                                     bd=0, highlightthickness=2,
                                     highlightbackground="#49644e", highlightcolor="#d9a24c",
                                     bg="#f5f0df", fg="#26372b", insertbackground="#26372b")
        self.request_text.pack(fill="x", ipady=12, ipadx=12)
        self.request_hint = "Ask for food ideas, check your dish, or correct/remove a label… (press Return)"
        self.request_text.insert(0, self.request_hint)
        self.request_text.bind("<FocusIn>", self._clear_request_hint)
        self.request_text.bind("<Return>", self._submit_request)
        self.root.after(300, self.request_text.focus_set)

    def _clear_request_hint(self, _event=None) -> None:
        if self.request_text.get() == self.request_hint:
            self.request_text.delete(0, "end")
            self.request_text.config(fg="#26372b")

    def _capture_gemini_image(self) -> bytes | None:
        """Encode an unmirrored, native-resolution camera frame for Gemini."""
        if not self.latest:
            return None
        frame = self.latest.get("source_frame", self.latest.get("frame"))
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            return None
        print(f"[Gemini] Captured native-resolution image: {frame.shape[1]}×{frame.shape[0]}.", flush=True)
        return encoded.tobytes()

    def _capture_gemini_crops(self) -> list[tuple[str, bytes]]:
        """Attach high-resolution crops for visible foods and unreviewed proposals."""
        if not self.latest or "source_frame" not in self.latest:
            return []
        raw_frame = self.latest["source_frame"]
        items = list(self._visible_items()) + self._select_audit_candidates()
        unique: list[dict] = []
        seen_ids: set[int] = set()
        for item in items:
            track_id = item.get("id")
            if track_id is not None and track_id in seen_ids:
                continue
            if any(self._box_iou(item["bbox"], previous["bbox"]) > 0.62 for previous in unique):
                continue
            unique.append(item)
            if track_id is not None:
                seen_ids.add(track_id)
            if len(unique) >= MAX_GEMINI_CANDIDATES:
                break
        crops: list[tuple[str, bytes]] = []
        for index, item in enumerate(unique, start=1):
            label = (
                self._display_name(item)
                if item.get("id") in self.confirmed_candidate_track_ids
                else f"unclassified {item.get('model_label', item['name'])} candidate"
            )
            crop = self._candidate_crop(raw_frame, item["bbox"])
            if crop:
                crops.append((f"Image crop #{index}: {label}.", crop))
        return crops

    def _submit_request(self, _event=None):
        request = self.request_text.get().strip()
        if not request or request == self.request_hint or self.gemini_busy:
            return "break"

        is_progress_check = self._looks_like_progress_check(request)
        is_cooking_check = self._looks_like_cooking_check(request)
        is_recipe_request = self._looks_like_recipe_request(request)
        if self._apply_label_removal(request):
            self.request_text.delete(0, "end")
            return "break"
        if self._looks_like_removal(request):
            visible_names = sorted({self._display_name(item).casefold() for item in self._visible_items()})
            message = (
                "No visible food label is available to remove."
                if not visible_names
                else "Could not match that removal request. Visible labels: " + ", ".join(visible_names)
            )
            self._set_status_note(message)
            print(f"[Labels] {message}", flush=True)
            self.request_text.delete(0, "end")
            return "break"
        explicit_correction = re.match(
            r"^(?:please\s+)?(?:correct|change|rename|relabel|fix|label)\b|"
            r"^(?:i meant|no[, ]|not\s+the\s+)",
            request, re.I,
        )
        is_visual_check = is_progress_check or is_cooking_check
        if (not is_visual_check or explicit_correction) and self._apply_label_correction(request):
            self.request_text.delete(0, "end")
            return "break"
        if (not is_visual_check or explicit_correction) and self._looks_like_correction(request):
            visible_names = sorted(
                {self._display_name(item).casefold() for item in self._visible_items()}
            )
            if not visible_names:
                message = "No food label is visible to correct. The detector must find the item first."
            else:
                message = "Correction not applied. Use “change old label to new label.” Visible labels: " + ", ".join(visible_names)
            self._set_status_note(message)
            print(f"[Labels] {message}", flush=True)
            self.request_text.delete(0, "end")
            return "break"

        foods = list(dict.fromkeys(self._display_name(item) for item in self._visible_items()))
        ingredients = ", ".join(foods) if foods else "No food items are currently detected."
        image_bytes = None
        image_crops: list[tuple[str, bytes]] = []
        if is_visual_check or is_recipe_request:
            image_bytes = self._capture_gemini_image()
            if image_bytes is None:
                self._set_status_note("Camera is not ready yet. Try that question again in a moment.")
                return "break"
            image_crops = self._capture_gemini_crops()
            print(f"[Gemini] Request includes {len(image_crops)} high-resolution item crop(s).", flush=True)
            if is_progress_check:
                prompt = (
                    "Assess the user's cooking progress from the attached native-resolution camera photo and item close-ups. "
                    "Compare what is visibly happening with the recent recipe or cooking context "
                    "when available. In under 100 words, say what stage the food appears to be at, "
                    "whether it seems on track (with a visible reason), and the single best next "
                    "step. Mention a useful parallel prep task only if it genuinely saves time. "
                    "Do not invent elapsed time, completion percentages, ingredients, or hidden "
                    "actions. If there is no recipe context or the photo is unclear, say what you "
                    "can and cannot tell, then give a cautious practical next step. A photo cannot "
                    "establish food safety or internal doneness; for meat, poultry, or egg safety, "
                    "recommend checking with a food thermometer.\n"
                    f"Possibly detected foods (may be wrong): {ingredients}\n"
                    f"Recent recipe/cooking context: {self.last_recipe_context or 'None provided.'}\n"
                    f"User's progress question: {request}"
                )
                print("[Input] Cooking progress check detected; sending the latest camera photo and recipe context to Gemini.", flush=True)
            elif is_cooking_check:
                prompt = (
                    "Look at the attached native-resolution camera photo and item close-ups, then answer the user's cooking check. "
                    "Be concise (under 90 words). Say what food and visible cooking stage you can "
                    "identify, whether it appears burnt, underdone, overdone, or about right for "
                    "the user's stated goal, and cite visible cues. Give one practical next step, "
                    "or say it looks fine as-is. Do not guess when the photo is unclear. A photo "
                    "cannot establish food safety or internal doneness; if the question concerns "
                    "meat, poultry, or egg safety, say to check with a food thermometer.\n"
                    f"Possibly detected foods (may be wrong): {ingredients}\n"
                    f"Recent recipe/cooking context: {self.last_recipe_context or 'None provided.'}\n"
                    f"User's question: {request}"
                )
                print("[Input] Cooking check detected; sending the latest camera photo and question to Gemini.", flush=True)
            else:
                prompt = (
                    "Inspect the attached current camera photo as well as the local detections. "
                    "Use the separate close-up crops to inspect small items and package text. "
                    "Suggest up to 3 concise recipe ideas using only foods or ingredients you can "
                    "clearly identify. Treat labels and packages carefully; do not assume an item "
                    "is milk, cheese, or another ingredient from container shape alone. Briefly note "
                    "key missing ingredients. If the image is unclear, say so. Stay under 80 words.\n"
                    f"Possibly detected foods: {ingredients}\nRequest: {request}"
                )
            if is_recipe_request:
                self.last_recipe_context = f"User's recipe request: {request}"
                self.last_operation_kind = "recipe"
                print("[Input] Sending recipe request, native-resolution camera photo, and detections to Gemini.", flush=True)
        else:
            prompt = (
                "Answer the user's general question directly and concisely. Use detected-food "
                "context only when relevant. Do not turn the request into a recipe unless the "
                "user asks for one. Stay under 100 words.\n"
                f"Detected foods: {ingredients}\nUser's question: {request}"
            )
            self.last_operation_kind = "general"
            print("[Input] Sending general question to Gemini.", flush=True)
        if is_progress_check:
            self.last_operation_kind = "progress_check"
        elif is_cooking_check:
            self.last_operation_kind = "cooking_check"
        self.gemini_busy = True
        self.pending_ai_question = request
        self._append_user_message(request)
        self.request_text.config(state="disabled")
        self.status.config(
            text=("Checking cooking progress…" if is_progress_check else
                  "Checking dish with Gemini…" if is_cooking_check else "Asking Gemini…")
        )
        threading.Thread(
            target=self._ask_gemini, args=(prompt, image_bytes, image_crops), daemon=True
        ).start()
        return "break"

    def _looks_like_removal(self, text: str) -> bool:
        normalized = text.strip().replace("’", "'")
        return bool(re.match(
            r"^(?:please\s+)?(?:remove|hide|dismiss|ignore|delete|clear|stop showing|don't show|do not show)\b|"
            r"^(?:please\s+)?get\s+rid\s+of\b|"
            r"^(?:(?:this|that|it|it's|that's)\s+)?(?:is\s+)?(?:not|isn't|is\s+not)\s+"
            r"(?:a\s+)?(?:food|food item|ingredient|edible)\b|"
            r"^(?:i\s+)?(?:don't|do not)\s+want\s+(?:this|that|it|the label)\b",
            normalized, re.I,
        ))

    def _apply_label_removal(self, text: str) -> bool:
        """Hide a false-positive detection for the current tracked object(s)."""
        items = self._visible_items()
        if not items:
            return False
        normalized = text.strip().replace("’", "'")
        targets = []
        visible_names = sorted(
            {self._display_name(item).casefold() for item in items}, key=len, reverse=True
        )
        remove_verbs = r"(?:remove|hide|dismiss|ignore|delete|clear|get rid of|stop showing|don't show|do not show)"
        for name in visible_names:
            escaped = re.escape(name)
            if re.match(
                rf"^(?:please\s+)?{remove_verbs}\s+(?:the\s+)?{escaped}"
                rf"(?:\s+(?:label|box|from (?:the )?(?:screen|view)))?[.! ]*$|"
                rf"^(?:please\s+)?(?:remove|hide|delete|stop showing|don't show|do not show)\s+(?:the\s+)?(?:label|box)\s+for\s+(?:the\s+)?{escaped}[.! ]*$",
                normalized, re.I,
            ):
                targets = [item for item in items if self._display_name(item).casefold() == name]
                break
        if not targets and len(items) == 1 and re.match(
            rf"^(?:please\s+)?{remove_verbs}\s+(?:this|that|it|the label|the box|this label|that label)|"
            r"^(?:(?:this|that|it|it's|that's)\s+)?(?:is\s+)?(?:not|isn't|is\s+not)\s+"
            r"(?:a\s+)?(?:food|food item|ingredient|edible)\b|"
            r"^(?:i\s+)?(?:don't|do not)\s+want\s+(?:this|that|it|the label)\b",
            normalized, re.I,
        ):
            targets = items
        if not targets:
            return False
        names = []
        for item in targets:
            name = self._display_name(item)
            names.append(name)
            track_id = item.get("id")
            if track_id is None:
                self.ignored_class_names.add(item["name"].casefold())
            else:
                self.ignored_track_ids.add(track_id)
        message = "Removed label: " + ", ".join(dict.fromkeys(names))
        self._set_status_note(message)
        print(f"[Labels] {message} (local only).", flush=True)
        return True

    @staticmethod
    def _looks_like_progress_check(text: str) -> bool:
        """Route cooking-session progress questions to Gemini with the current camera photo."""
        normalized = re.sub(r"\s+", " ", text.lower().replace("’", "'"))
        phrases = (
            r"\bhow(?:'s| is) (?:it|this|the cooking|the dish|the recipe) going\b",
            r"\bhow are (?:we|things) doing\b",
            r"\b(?:am i|are we|is it) on track\b",
            r"\bhow far along (?:am i|are we|is it|are we in the recipe)\b",
            r"\bwhat stage (?:am i|are we|is this|is the dish|is the recipe) at\b",
            r"\bwhere am i in (?:the )?(?:recipe|cooking|process|steps?)\b",
            r"\bwhat(?:'s| is) (?:my|the) next step\b",
            r"\bwhat should i do next\b",
            r"\bwhat do i do (?:now|next)\b",
            r"\bwhat(?:'s| is) next\b",
            r"\b(?:am i|are we) doing (?:this|it|the recipe) right\b",
            r"\b(?:am i|are we) doing the right thing\b",
            r"\b(?:is this|are we) going as planned\b",
            r"\bhow(?:'s| is) (?:the )?progress\b",
            r"\bwhat can i (?:prep|do) while .{1,50}\b(?:cooks?|cooking|heats?|heating|boils?|boiling)\b",
            r"\bcan i prep .{1,50} while\b",
        )
        return any(re.search(phrase, normalized) for phrase in phrases)

    @staticmethod
    def _looks_like_cooking_check(text: str) -> bool:
        """Route common appearance and doneness questions to vision."""
        normalized = re.sub(r"\s+", " ", text.lower().replace("’", "'"))
        cooking_terms = (
            r"\b(cook(?:ed|ing)?|burnt|burned|overcook(?:ed)?|undercook(?:ed)?|"
            r"doneness|ready|done|raw|crispy|crisp|golden|browned|underdone|overdone|"
            r"burn|burning|smoking|smoky|"
            r"soggy|too salty|too dry|too oily)\b"
        )
        appearance_phrases = (
            r"\b(?:how does|how do|how's|how is)\s+(?:this|it|that|the dish|the food)\s+look\b",
            r"\blook(?:s|ing)?\s+(?:good|right|tasty|okay|ok|fine|burnt|burned|done|cooked|ready|appetizing|edible)\b",
            r"\b(?:is this|does this|does it|is it)\s+(?:right|okay|ok|good|tasty|fine)\b",
            r"\bwhat do you think (?:of|about) (?:this|it|the dish|the food)\b",
            r"\bhow(?:'s| is) (?:this|it|the dish) coming along\b",
            r"\bhow(?:'s| is) it looking\b",
            r"\b(?:is|does|will|would|how)\b.{0,35}\b(?:this|it|that|dish|food|meal)\b.{0,35}\b(?:look|taste|seem|appear|right|good|fine)\b",
            r"\b(?:appearance|too dark|too brown|getting brown|turning brown)\b",
            r"\b(?:is|does)\s+(?:this|it|that|the dish|the food)\s+(?:cooked|done|ready|burnt|burned|raw|safe)\b",
            r"\b(?:can i eat|is it okay to eat|is this okay to eat)\b",
            r"\b(?:does this|do these|is this|are these) (?:look|taste|seem|feel) (?:okay|ok|right|good)\b",
            r"\b(?:did i|have i) (?:burn|overcook|undercook|ruin)\b",
            r"\b(?:do i need to|should i) (?:flip|stir|turn|cover|lower|raise|add|remove)\b",
            r"\b(?:can i|how do i) (?:save|fix) (?:this|it|the dish)\b",
            r"\b(?:is|are) .{1,30} (?:too salty|too dry|too oily|too soggy)\b",
            r"\bwhat should i do next\b",
            r"\bwhat (?:do i do|should i do) now\b",
            r"\bwhat(?:'s| is) wrong with (?:this|it|the dish)\b",
            r"\bany tips to (?:improve|fix|save) (?:this|it|the dish)\b",
            r"\bwhat(?:'s| is) next\b",
            r"\bhow can i (?:fix|save|improve) (?:this|it|the dish)\b",
            r"\bshould i (?:flip|stir|remove|take|turn|cover|lower|raise|add)\b",
            r"\b(?:take|remove) (?:it|this|the dish) off (?:the )?(?:heat|stove)\b",
        )
        return bool(re.search(cooking_terms, normalized) or any(
            re.search(pattern, normalized) for pattern in appearance_phrases
        ))

    @staticmethod
    def _looks_like_recipe_request(text: str) -> bool:
        """Recognize meal-idea requests; unmatched prompts use general Gemini chat."""
        normalized = re.sub(r"\s+", " ", text.lower().replace("’", "'"))
        patterns = (
            r"\b(?:recipe|recipes|meal ideas?|dish ideas?|snack ideas?|dinner ideas?|lunch ideas?|breakfast ideas?)\b",
            r"\bwhat can i (?:make|cook|prepare|eat)\b",
            r"\bwhat should i (?:make|cook|eat)\b",
            r"\bhelp me (?:make|cook|prepare)\b",
            r"\bi need (?:a|some|an?) (?:quick |easy )?(?:recipe|meal|dinner|lunch|breakfast|snack)\b",
            r"\bwhat(?:'s| is) (?:a good|something easy|something quick) (?:thing|meal|dish|recipe) to (?:make|cook|eat)\b",
            r"\b(?:give|suggest|recommend|find|show) me (?:a |some |an )?(?:recipe|recipes|meal|dish|ideas?)\b",
            r"\bhow (?:can|do) i (?:make|cook|prepare)\b",
            r"\bhow (?:can|do) i use (?:these |the |my )?ingredients\b",
            r"\b(?:make|cook|prepare) (?:me )?something (?:with|using)\b",
            r"\bideas? (?:for|using|with)\b",
            r"\b(?:i want|i'd like|i would like) (?:to eat|a meal|something to eat)\b",
            r"\b(?:i'm|i am) hungry\b",
            r"\b(?:use|using) (?:the |my )?(?:ingredients|food|leftovers)\b",
            r"\bwhat goes well with\b",
            r"\bwhat can .* ingredients\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _run_initial_label_audit(self) -> None:
        """Review the first stable set of food and object proposals after camera startup."""
        self._maybe_review_new_objects(force=True)

    @staticmethod
    def _box_iou(first: list[int], second: list[int]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        width = max(0, min(ax2, bx2) - max(ax1, bx1))
        height = max(0, min(ay2, by2) - max(ay1, by1))
        intersection = width * height
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return intersection / max(1, area_a + area_b - intersection)

    def _select_audit_candidates(self) -> list[dict]:
        """Collect stable, unreviewed food guesses and generic object proposals."""
        if not self.latest or "detections" not in self.latest:
            return []
        frame_height, frame_width = self.latest["frame"].shape[:2]
        frame_area = max(1, frame_height * frame_width)
        detections = [
            item for item in self.latest["detections"]
            if item.get("id") is not None
            and item["id"] not in self.reviewed_track_ids
            and item["id"] not in self.ignored_track_ids
            and item["id"] not in self.track_label_overrides
            and (
                item.get("kind") == "food"
                or (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])
                < frame_area * 0.75
            )
        ]
        detections.sort(key=lambda item: (item.get("kind") != "food", -item["confidence"]))
        selected: list[dict] = []
        food_count = 0
        for item in detections:
            if item.get("kind") == "food" and food_count >= MAX_FOOD_AUDIT_CANDIDATES:
                continue
            if any(self._box_iou(item["bbox"], prior["bbox"]) > 0.62 for prior in selected):
                continue
            selected.append(item)
            if item.get("kind") == "food":
                food_count += 1
            if len(selected) >= MAX_GEMINI_CANDIDATES:
                break
        return selected

    @staticmethod
    def _candidate_crop(raw_frame: np.ndarray, display_bbox: list[int]) -> bytes | None:
        """Crop the unmirrored source using a mirrored display box, preserving detail."""
        height, width = raw_frame.shape[:2]
        dx1, dy1, dx2, dy2 = display_bbox
        x1, x2 = width - dx2, width - dx1
        box_width, box_height = max(1, x2 - x1), max(1, dy2 - dy1)
        pad_x, pad_y = round(box_width * 0.16), round(box_height * 0.16)
        x1, x2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
        y1, y2 = max(0, dy1 - pad_y), min(height, dy2 + pad_y)
        crop = raw_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        longest = max(crop.shape[:2])
        if longest < 700:
            scale = 700 / longest
            crop = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)),
                              interpolation=cv2.INTER_CUBIC)
        elif longest > 1400:
            scale = 1400 / longest
            crop = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
        return encoded.tobytes() if ok else None

    def _maybe_review_new_objects(self, *, force: bool = False) -> None:
        if self.stop_event.is_set() or self.label_audit_in_flight:
            return
        now = time.monotonic()
        if not force and now - self.last_label_audit_at < LABEL_REVIEW_INTERVAL:
            return
        if not self.latest or "source_frame" not in self.latest:
            return
        items = self._select_audit_candidates()
        if not items:
            return

        raw_frame = self.latest["source_frame"].copy()
        ok, encoded = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            print("[Gemini] Could not encode the full-resolution camera frame.", flush=True)
            return
        references: dict[int, dict] = {}
        crop_images: list[tuple[int, bytes]] = []
        for reference, item in enumerate(items, start=1):
            references[reference] = {
                "id": item["id"], "name": item["name"],
                "kind": item["kind"], "bbox": item["bbox"],
            }
            crop = self._candidate_crop(raw_frame, item["bbox"])
            if crop:
                crop_images.append((reference, crop))

        self.label_audit_in_flight = True
        self.last_label_audit_at = now
        self.initial_audit_started = True
        self._set_status_note("Gemini is checking the camera items…")
        print(
            f"[Gemini] Sending a {raw_frame.shape[1]}×{raw_frame.shape[0]} full frame "
            f"and {len(crop_images)} high-resolution item crops for review.", flush=True,
        )
        threading.Thread(
            target=self._audit_labels_with_gemini,
            args=(encoded.tobytes(), crop_images, references),
            daemon=True,
        ).start()

    def _audit_labels_with_gemini(
        self, image_bytes: bytes, crop_images: list[tuple[int, bytes]], references: dict[int, dict]
    ) -> None:
        started = time.perf_counter()
        try:
            if not os.environ.get("GEMINI_API_KEY"):
                raise RuntimeError("GEMINI_API_KEY is not set. Set it in Terminal, then restart the app.")
            from google import genai
            from google.genai import types

            with self.gemini_client_lock:
                if self.gemini_client is None:
                    self.gemini_client = genai.Client()
                client = self.gemini_client
            reference_text = "; ".join(
                f"#{reference}: YOLO candidate '{item['name']}' ({item['kind']})"
                for reference, item in references.items()
            )
            prompt = (
                "You are checking camera detections for a cooking app. Inspect the full "
                "unmirrored kitchen image, then each corresponding numbered high-resolution crop. "
                "For every reference, decide whether it is food or a cooking ingredient, including "
                "sauces, condiments, oils, tortillas, cheese, seasonings, spices, herbs, flour, "
                "grains, canned or packaged foods, drinks, and other pantry items. These count as "
                "food even when they are not ready to eat by themselves. Read visible package text "
                "when possible and label the food or ingredient inside, not merely its container. "
                "Mark utensils, people, "
                "furniture, empty packaging, and non-food objects edible=false. If uncertain, "
                "return edible=false. Return exactly one item per reference; never invent a "
                "reference or bounding box.\n"
                f"References: {reference_text}\n"
                'Return strict JSON only: {"items":[{"reference":1,"edible":true,'
                '"label":"milk","confidence":"high"}]}. Include every reference. '
                'Confidence must be "high", "medium", or "low". For non-food or uncertain '
                'objects use edible=false, label="", confidence="low".'
            )
            high = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
            contents = [types.Part.from_bytes(
                data=image_bytes, mime_type="image/jpeg", media_resolution=high
            )]
            for reference, crop_bytes in crop_images:
                contents.append(types.Part.from_text(text=f"High-resolution crop for candidate #{reference}."))
                contents.append(types.Part.from_bytes(
                    data=crop_bytes, mime_type="image/jpeg", media_resolution=high
                ))
            contents.append(types.Part.from_text(text=prompt))
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=700,
                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                ),
            )
            result_text = (response.text or "{}").strip()
            result = json.loads(result_text)
            items = result.get("items", []) if isinstance(result, dict) else []
            if not isinstance(items, list):
                items = []
            self.audit_results.put((references, "ok", json.dumps({"items": items})))
            print(f"[Gemini] Item review returned in {time.perf_counter() - started:.1f}s.", flush=True)
        except Exception as error:
            self.audit_results.put((references, "error", str(error)))
            print(f"[Gemini] Initial label check failed: {error}", flush=True)

    def _looks_like_correction(self, text: str) -> bool:
        if re.match(
            r"^(?:please\s+)?(?:correct|change|rename|relabel|fix|label|call)\b|"
            r"^(?:this is|that is|that's|it is|it's|actually|i meant|no[, ])\b|"
            r"^not\s+(?:the\s+)?\w+[, ]",
            text.strip(), re.I,
        ):
            return True
        # Treat “cake is a chips box” as a correction even if applying it fails,
        # so it can never silently become a Gemini recipe request.
        known_names = sorted(
            set(FOOD_CLASSES) | {self._display_name(item) for item in self._visible_items()},
            key=len, reverse=True,
        )
        for name in known_names:
            pattern = (
                rf"^(?:the\s+)?{re.escape(name)}\s+"
                rf"(?:is|is actually|should be|should be called|should say|was mislabeled as|"
                rf"isn't|is not)\s+.+$"
            )
            if re.match(pattern, text.strip(), re.I):
                return True
        return False

    def _set_status_note(self, message: str) -> None:
        self.status_note = message
        self.status_note_until = time.monotonic() + 3.0
        self.status.config(text=message)

    @staticmethod
    def _clean_label(value: str) -> str:
        value = re.sub(r"^(?:a|an|the)\s+", "", value.strip(), flags=re.I)
        return value.strip(" .,!?'\\\";").lower()

    def _display_name(self, item: dict) -> str:
        track_id = item.get("id")
        if track_id is not None and track_id in self.track_label_overrides:
            return self.track_label_overrides[track_id]
        original = item["name"]
        return self.class_label_overrides.get(original.casefold(), original)

    def _apply_label_correction(self, text: str) -> bool:
        """Apply simple spoken corrections to the current detection labels."""
        items = self._visible_items()
        if not items:
            return False
        visible_names = sorted(
            {self._display_name(item).casefold() for item in items}, key=len, reverse=True
        )
        normalized = text.strip().replace("’", "'").replace("‘", "'")
        correction = None

        for old_name in visible_names:
            old = re.escape(old_name)
            patterns = (
                rf"^(?:please\s+)?(?:change|rename|correct)\s+(?:the\s+)?{old}(?:\s+label)?\s+(?:to|as)\s+(.+)$",
                rf"^(?:please\s+)?(?:change|correct|fix)\s+(?:the\s+)?(?:label\s+)?from\s+{old}\s+to\s+(.+)$",
                rf"^(?:please\s+)?label\s+(?:the\s+)?{old}\s+as\s+(.+)$",
                rf"^(?:(?:the|that|this)\s+)?{old}\s+(?:is actually|is|should be called|should be|should say|was mislabeled as)\s+(.+)$",
                rf"^(?:(?:the|that|this)\s+)?{old}\s+(?:isn't|is not)\s+(?:the\s+)?(.+)$",
                rf"^(?:not\s+(?:the\s+)?{old}[, ]+(?:it's|it is|this is)\s+)(.+)$",
                rf"^(?:the|that|this)\s+{old}\s+(?:label\s+)?(?:is wrong|is incorrect)[,;:]?\s*(?:it'?s|it is|this is)\s+(.+)$",
                rf"^(?:actually[, ]+)?(?:it'?s|it is|this is|that is)\s+(.+?),?\s+not\s+(?:the\s+)?{old}$",
                rf"^(?:it is|it's|that is|that's|actually)\s+(.+?),?\s+not\s+(?:the\s+)?{old}$",
            )
            for pattern in patterns:
                match = re.match(pattern, normalized, re.I)
                if match:
                    correction = (old_name, self._clean_label(match.group(1)))
                    break
            if correction:
                break

        # A bare “this is X” is safe only when there is exactly one visible box.
        if correction is None and len(items) == 1:
            match = re.match(
                r"^(?:(?:this|that|it)\s+(?:is|looks like|should be(?: called)?|is actually)|(?:it is|it's|that is|that's|no,?\s+(?:it's|this is|it is)|i meant|actually(?:,?\s+(?:it's|it is|this is))?))\s+(?:actually\s+)?(.+)$",
                normalized, re.I,
            )
            if not match:
                match = re.match(r"^(?:correct|label)\s+(?:this|that)\s+(?:as|to)\s+(.+)$",
                                 normalized, re.I)
            if not match:
                match = re.match(r"^call\s+(?:this|that|it)\s+(.+)$", normalized, re.I)
            if match:
                correction = (visible_names[0], self._clean_label(match.group(1)))

        if not correction or not correction[1]:
            return False
        old_name, new_name = correction
        matched = [item for item in items if self._display_name(item).casefold() == old_name]
        if not matched:
            return False
        for item in matched:
            track_id = item.get("id")
            if track_id is None:
                self.class_label_overrides[item["name"].casefold()] = new_name
            else:
                self.track_label_overrides[track_id] = new_name
        self._set_status_note(f"Label corrected: {old_name} → {new_name}")
        print(f"[Labels] Corrected {old_name} to {new_name} ({len(matched)} item(s)).", flush=True)
        return True

    def _ask_gemini(
        self, prompt: str, image_bytes: bytes | None = None,
        image_crops: list[tuple[str, bytes]] | None = None,
    ) -> None:
        started = time.perf_counter()
        progress_done = threading.Event()
        print("[Gemini] Preparing request…", flush=True)
        try:
            if not os.environ.get("GEMINI_API_KEY"):
                raise RuntimeError("GEMINI_API_KEY is not set. Set it in Terminal, then restart the app.")
            from google import genai
            from google.genai import types

            with self.gemini_client_lock:
                if self.gemini_client is None:
                    self.gemini_client = genai.Client()
                client = self.gemini_client
            print("[Gemini] Request sent; waiting for response…", flush=True)

            def report_waiting() -> None:
                while not progress_done.wait(2.0):
                    elapsed = time.perf_counter() - started
                    print(f"[Gemini] Still waiting ({elapsed:.0f}s)…", flush=True)

            threading.Thread(target=report_waiting, daemon=True).start()
            if image_bytes is not None:
                high = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
                contents = [
                    types.Part.from_bytes(
                        data=image_bytes, mime_type="image/jpeg", media_resolution=high
                    )
                ]
                for caption, crop_bytes in image_crops or []:
                    contents.append(types.Part.from_text(text=caption))
                    contents.append(types.Part.from_bytes(
                        data=crop_bytes, mime_type="image/jpeg", media_resolution=high
                    ))
                contents.append(types.Part.from_text(text=prompt))
            else:
                contents = [prompt]
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are Peter Cookin, an insanely enthusiastic, overconfident cooking companion "
                        "with the wild humor and personality of Peter Griffin from Family Guy. Speak in character "
                        "with high energy, hilarious overconfidence, and iconic catchphrases like 'Holy crap, bro!', "
                        "'Freakin' sweet, dude!', 'Whoa whoa whoa whoa whoa', 'WHY ARE WE NOT FUNDING THIS?!', "
                        "'You know what really grinds my gears?', 'A boat's a boat, but the mystery box could be "
                        "anything!', 'I may be an idiot, but there is one thing I am not, sir, and that is an idiot!', "
                        "or 'This reminds me of that time I...'. Always include his signature laugh explicitly written "
                        "out as 'hehehehehe' mid-sentence or at the end of thoughts. Keep responses concise in two "
                        "or three sentences total and use plain conversational text only: write out all numbers as "
                        "full words and use zero Markdown, asterisks, bold marks, headings, tables, or bullet symbols "
                        "so text-to-speech audio reads smoothly. Keep uncertainty and food-safety guidance crystal "
                        "clear; never trade accuracy for enthusiasm or comedy."
                    ),
                    media_resolution=(
                        types.MediaResolution.MEDIA_RESOLUTION_HIGH if image_bytes is not None else None
                    ),
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                ),
            )
            answer = (response.text or "Gemini returned an empty response.").strip()
            print(f"[Gemini] Response received in {time.perf_counter() - started:.1f}s.", flush=True)
            self.gemini_results.put(("ok", answer))
        except Exception as error:
            print(f"[Gemini] Request failed after {time.perf_counter() - started:.1f}s: {error}", flush=True)
            self.gemini_results.put(("error", str(error)))
        finally:
            progress_done.set()

    def _speech_loop(self) -> None:
        """Serialize speech playback off the camera and Tkinter UI threads."""
        while not self.stop_event.is_set():
            try:
                response_text = self.speech_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                print("[TTS] Generating speech for Gemini's reply…", flush=True)
                speak_response(response_text)
                print("[TTS] Playback finished.", flush=True)
            except Exception as error:
                print(f"[TTS] Could not speak this reply: {error}", flush=True)

    def _append_user_message(self, question: str) -> None:
        """Show the submitted prompt immediately in the persistent AI sidebar."""
        self.ai_chat.configure(state="normal")
        self.ai_chat.insert("end", "\n\nYOU\n", "you")
        self.ai_chat.insert("end", f"{question}\n", "message")
        self.ai_chat.see("end")
        self.ai_chat.configure(state="disabled")
        self.ai_chat.update_idletasks()

    def _append_ai_response(self, response_text: str, *, error: bool = False) -> None:
        """Append Gemini's response beneath the already-visible user prompt."""
        self.ai_chat.configure(state="normal")
        self.ai_chat.insert("end", "\nPETER COOKIN\n" if not error else "\nREQUEST ISSUE\n",
                            "peter" if not error else "error")
        self.ai_chat.insert("end", f"{response_text}\n", "message" if not error else "error")
        self.ai_chat.see("end")
        self.ai_chat.configure(state="disabled")

    def _camera_loop(self) -> None:
        """Capture full-resolution preview frames without waiting for detector inference."""
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            self._publish({"error": "Could not open camera. Check camera permission for Terminal."})
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, 60)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        requested_fps = camera.get(cv2.CAP_PROP_FPS)
        print(
            f"[Camera] Preview capture: {actual_width}x{actual_height}; "
            f"requested up to 60 FPS (device reports {requested_fps:.0f}); "
            f"detector image size remains {INFERENCE_SIZE}.", flush=True
        )
        last_time = time.perf_counter()
        fps = 0.0
        try:
            while not self.stop_event.is_set():
                ok, raw_frame = camera.read()
                if not ok:
                    self._publish({"error": "Camera frame could not be read."})
                    break
                now = time.perf_counter()
                elapsed = now - last_time
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    fps = instant_fps if fps == 0 else fps * 0.82 + instant_fps * 0.18
                last_time = now
                with self.preview_lock:
                    self.preview_frame = raw_frame
                    self.preview_sequence += 1
                    self.capture_fps = fps
                try:
                    if self.inference_frames.full():
                        self.inference_frames.get_nowait()
                    self.inference_frames.put_nowait(raw_frame)
                except (queue.Empty, queue.Full):
                    pass
        finally:
            camera.release()

    def _inference_loop(self) -> None:
        """Run detector on the newest captured frame, keeping its input size unchanged."""
        last_time = time.perf_counter()
        fps = 0.0
        while not self.stop_event.is_set():
            try:
                raw_frame = self.inference_frames.get(timeout=0.2)
            except queue.Empty:
                continue
            self.ocr.submit(raw_frame)
            package_text = self.ocr.latest_text()
            text_match = food_name_from_text(package_text)
            frame = cv2.flip(raw_frame, 1)
            results = self.model.track(
                frame, persist=True, tracker=str(TRACKER_CONFIG),
                conf=DETECTOR_CONFIDENCE, imgsz=INFERENCE_SIZE, verbose=False,
            )
            detections = []
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                tracked_ids = result.boxes.id
                ids = tracked_ids.int().cpu().tolist() if tracked_ids is not None else [None] * len(boxes)
                for coordinates, class_id, confidence, track_id in zip(boxes, class_ids, confidences, ids):
                    model_label = str(self.model.names[class_id]).casefold()
                    is_object_proposal = model_label in OBJECT_PROPOSAL_CLASS_SET
                    name = FOOD_CLASS_ALIASES.get(model_label, model_label)
                    confidence = float(confidence)
                    if not is_object_proposal and name == "cake" and confidence < 0.55:
                        if text_match and text_match != "cake":
                            name = text_match
                        elif text_match != "cake":
                            continue
                    elif not is_object_proposal and confidence < (
                        MILK_DISPLAY_CONFIDENCE
                        if name == "milk"
                        else LEAFY_DISPLAY_CONFIDENCE
                        if name in LOW_CONFIDENCE_LEAFY_FOODS
                        else DISPLAY_CONFIDENCE
                    ) and text_match != name:
                        continue
                    detections.append({
                        "name": name, "model_label": model_label,
                        "kind": "object_proposal" if is_object_proposal else "food",
                        "confidence": confidence, "id": track_id,
                        "bbox": [int(value) for value in coordinates],
                    })
            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else fps * 0.82 + instant_fps * 0.18
            last_time = now
            self._publish({"frame": frame, "source_frame": raw_frame,
                           "detections": detections, "fps": fps})

    def _publish(self, payload: dict) -> None:
        try:
            if self.frames.full():
                self.frames.get_nowait()
            self.frames.put_nowait(payload)
        except (queue.Empty, queue.Full):
            pass

    def _refresh(self) -> None:
        refresh_started = time.perf_counter()
        try:
            while True:
                self.latest = self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                result_type, result_text = self.gemini_results.get_nowait()
                self.gemini_busy = False
                self.request_text.config(state="normal")
                if result_type == "ok":
                    if self.last_operation_kind == "recipe":
                        self.last_recipe_context += f"\nGemini's recipe suggestions: {result_text}"
                    elif self.last_operation_kind == "progress_check":
                        self.last_recipe_context = (
                            f"{self.last_recipe_context}\nLatest cooking progress update: {result_text}"
                        )[-4000:]
                    self.status.config(text="Gemini response ready")
                    self._append_ai_response(result_text)
                    if self.tts_enabled:
                        try:
                            self.speech_queue.put_nowait(result_text)
                            print("[TTS] Gemini reply queued for the configured ElevenLabs voice.", flush=True)
                        except queue.Full:
                            print("[TTS] Speech queue is full; skipped this reply.", flush=True)
                    else:
                        print("[TTS] Gemini reply is text-only because ElevenLabs credentials are unavailable.", flush=True)
                else:
                    self.status.config(text="Gemini request failed")
                    self._append_ai_response(result_text, error=True)
                if self.request_text.get() == self.pending_ai_question:
                    self.request_text.delete(0, "end")
                self.pending_ai_question = ""
        except queue.Empty:
            pass
        try:
            while True:
                references, result_type, result_text = self.audit_results.get_nowait()
                self.label_audit_in_flight = False
                if result_type == "error":
                    self._set_status_note("Gemini item check failed; will retry.")
                    continue
                try:
                    audit_result = json.loads(result_text)
                except json.JSONDecodeError:
                    audit_result = {}
                decisions = audit_result.get("items", []) if isinstance(audit_result, dict) else []
                decisions_by_reference = {
                    item.get("reference"): item for item in decisions
                    if isinstance(item, dict) and isinstance(item.get("reference"), int)
                } if isinstance(decisions, list) else {}
                current_detections = self.latest.get("detections", []) if self.latest else []
                accepted = 0
                discarded = 0
                for reference, original in references.items():
                    original_id = original["id"]
                    decision = decisions_by_reference.get(reference, {})
                    edible = decision.get("edible") is True
                    confidence = str(decision.get("confidence", "")).casefold()
                    label = self._clean_label(decision.get("label", "")) if isinstance(decision.get("label"), str) else ""
                    if not edible or confidence not in {"high", "medium"} or not label or len(label) > 40:
                        self.reviewed_track_ids.add(original_id)
                        self.ignored_track_ids.add(original_id)
                        discarded += 1
                        continue

                    current = next(
                        (item for item in current_detections if item.get("id") == original_id), None
                    )
                    if current is None:
                        current = max(
                            current_detections,
                            key=lambda item: self._box_iou(item["bbox"], original["bbox"]),
                            default=None,
                        )
                        if current is not None and self._box_iou(current["bbox"], original["bbox"]) < 0.30:
                            current = None
                    if current is None:
                        self.reviewed_track_ids.add(original_id)
                        continue
                    track_id = current.get("id")
                    if track_id is None or track_id in self.track_label_overrides:
                        continue
                    self.reviewed_track_ids.update((original_id, track_id))
                    self.ignored_track_ids.discard(track_id)
                    self.track_label_overrides[track_id] = label
                    self.confirmed_candidate_track_ids.add(track_id)
                    accepted += 1
                    print(f"[Gemini] Confirmed edible item: {label} (track {track_id}).", flush=True)
                message = (
                    f"Gemini item check complete: labeled {accepted} edible item(s); "
                    f"discarded {discarded} other candidate(s)."
                )
                print(f"[Gemini] {message}", flush=True)
                self._set_status_note(message)
        except queue.Empty:
            pass
        self._maybe_review_new_objects()
        if self.latest:
            if "error" in self.latest:
                self.status.config(text=self.latest["error"])
            else:
                if time.monotonic() < self.status_note_until:
                    self.status.config(text=self.status_note)
                else:
                    with self.preview_lock:
                        preview_fps = self.capture_fps
                    self.status.config(text="Live")
        self._render_camera()
        if not self.stop_event.is_set():
            refresh_ms = max(1, int((1 / 60 - (time.perf_counter() - refresh_started)) * 1000))
            self.root.after(refresh_ms, self._refresh)

    def _visible_items(self) -> list[dict]:
        if not self.latest or "detections" not in self.latest:
            return []
        return [
            item for item in self.latest["detections"]
            if item.get("id") not in self.ignored_track_ids
            and item["name"].casefold() not in self.ignored_class_names
            and item.get("id") in self.confirmed_candidate_track_ids
        ]

    def _render_camera(self) -> None:
        with self.preview_lock:
            if self.preview_frame is None or self.preview_sequence == self.last_rendered_sequence:
                return
            frame = cv2.flip(self.preview_frame, 1)
            sequence = self.preview_sequence
        now = time.perf_counter()
        if self.last_displayed_frame_at:
            elapsed = now - self.last_displayed_frame_at
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                self.display_fps = (
                    instant_fps if self.display_fps == 0
                    else self.display_fps * 0.82 + instant_fps * 0.18
                )
        self.last_displayed_frame_at = now
        self.last_rendered_sequence = sequence
        for item in self._visible_items():
            draw_label(frame, tuple(item["bbox"]),
                       f"{self._display_name(item)}  {item['confidence']:.0%}")
        target_width = max(1, self.camera_view.winfo_width())
        target_height = max(1, self.camera_view.winfo_height())
        scale = min(target_width / frame.shape[1], target_height / frame.shape[0])
        size = (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image_ref = ImageTk.PhotoImage(Image.fromarray(frame))
        self.camera_view.configure(image=self.image_ref, text="")

    def close(self) -> None:
        self.stop_event.set()
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception:
            pass
        self.ocr.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    print("Loading pretrained YOLOE model and food/object prompts…")
    print("The first launch downloads YOLOE weights and its text encoder; later launches use local copies.")
    try:
        model = YOLOE(MODEL_NAME)
        model.set_classes(MODEL_CLASSES)
    except Exception as error:
        legacy_weights = Path(__file__).with_name("yolov8s-worldv2.pt")
        if not legacy_weights.exists():
            raise RuntimeError(
                "YOLOE could not load. Connect to the internet for its first download, "
                "or restore the local YOLO-World weights."
            ) from error
        print(f"[Model] YOLOE unavailable ({error}); falling back to the local YOLO-World weights.", flush=True)
        model = YOLO(str(legacy_weights))
        # The fallback keeps the prior working detector available offline.
        model.set_classes(MODEL_CLASSES)
    app = PantryApp(model)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
