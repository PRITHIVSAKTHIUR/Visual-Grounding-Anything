# Dynamic Proximity-Based Point Association
# Pass 1: Dual-Tier Point Extraction (The Detector)
# Pass 2: Resolution-Invariant Distance Matching (The Primary Tracker)
# Pass 3: Temporal Track Patience (Flicker Handler)

# Install the required dependencies before running this script:
# pip install torch torchvision
# pip install gradio==6.9.0
# pip install transformers==5.3.0
# pip install opencv-python==4.13.0.92

import gc
import tempfile
import re
import json
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID_V = "prithivMLmods/Polaris-VGA-4B-Post1.0e"
DTYPE = torch.bfloat16

print(f"Loading {MODEL_ID_V}...")
processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
model_v = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID_V,
    trust_remote_code=True,
    torch_dtype=DTYPE
).to(device).eval()
print("Model loaded successfully.")

DEFAULT_MAX_SECONDS = 3.0
MAX_SECONDS_LIMIT = 20.0

SYSTEM_PROMPT = """You are a helpful assistant to detect objects in images. When asked to detect elements based on a description you return bounding boxes for all elements in the form of [xmin, ymin, xmax, ymax] with the values being scaled between 0 and 1000. When there are more than one result, answer with a list of bounding boxes in the form of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]."""

POINT_SYSTEM_PROMPT = """You are a precise object pointing assistant. When asked to point to an object in an image, you must return ONLY the exact center coordinates of that specific object as [x, y] with values scaled between 0 and 1000 (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner).

Rules:
1. ONLY point to objects that exactly match the description given.
2. Do NOT point to background, empty areas, or unrelated objects.
3. If there are multiple matching instances, return [[x1, y1], [x2, y2], ...].
4. If no matching object is found, return an empty list [].
5. Return ONLY the coordinate numbers, no explanations or other text.
6. Be extremely precise — place the point at the exact visual center of each matching object."""

POINTS_REGEX = re.compile(r'(?:(\d+)\s*[.:])?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)')


def try_load_video_frames(video_path_or_url: str):
    cap = cv2.VideoCapture(video_path_or_url)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, {"num_frames": len(frames), "fps": float(fps_val) if fps_val > 0 else None}


def parse_bboxes_from_text(text: str):
    text = re.sub(r'<think>.*?</think>', '', text.strip(), flags=re.DOTALL)

    nested = re.findall(r'\[\s*\[[\d\s,\.]+\](?:\s*,\s*\[[\d\s,\.]+\])*\s*\]', text)
    if nested:
        try:
            all_b = []
            for m in nested:
                parsed = json.loads(m)
                all_b.extend(parsed if isinstance(parsed[0], list) else [parsed])
            return all_b
        except (json.JSONDecodeError, IndexError):
            pass

    single = re.findall(
        r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
        text
    )
    if single:
        return [[float(v) for v in m] for m in single]

    nums = re.findall(r'(\d+(?:\.\d+)?)', text)
    return [
        [float(nums[i]), float(nums[i + 1]), float(nums[i + 2]), float(nums[i + 3])]
        for i in range(0, len(nums) - 3, 4)
    ] if len(nums) >= 4 else []


def parse_precise_points(text: str, image_w: int, image_h: int):
    text = re.sub(r'<think>.*?</think>', '', text.strip(), flags=re.DOTALL)
    raw_points = []

    nested = re.findall(r'\[\s*\[[\d\s,\.]+\](?:\s*,\s*\[[\d\s,\.]+\])*\s*\]', text)
    if nested:
        try:
            for m in nested:
                parsed = json.loads(m)
                if isinstance(parsed[0], list):
                    for p in parsed:
                        if len(p) >= 2:
                            raw_points.append((float(p[0]), float(p[1])))
                elif len(parsed) >= 2:
                    raw_points.append((float(parsed[0]), float(parsed[1])))
        except (json.JSONDecodeError, IndexError):
            pass

    if not raw_points:
        single = re.findall(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
        if single:
            for m in single:
                raw_points.append((float(m[0]), float(m[1])))

    if not raw_points:
        for match in POINTS_REGEX.finditer(text):
            x_val = float(match.group(2))
            y_val = float(match.group(3))
            raw_points.append((x_val, y_val))

    validated = []
    for sx, sy in raw_points:
        if not (0 <= sx <= 1000 and 0 <= sy <= 1000):
            continue
        px = sx / 1000 * image_w
        py = sy / 1000 * image_h
        if 0 <= px <= image_w and 0 <= py <= image_h:
            validated.append((px, py))

    if len(validated) > 1:
        deduped = [validated[0]]
        for pt in validated[1:]:
            is_dup = False
            for existing in deduped:
                dist = ((pt[0] - existing[0]) ** 2 + (pt[1] - existing[1]) ** 2) ** 0.5
                if dist < 15:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(pt)
        validated = deduped

    return validated


def pixel_point_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class PointTrackerState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.video_frames = []
        self.video_fps = None
        self.points_by_frame = {}
        self.trails = []

    @property
    def num_frames(self):
        return len(self.video_frames)


def detect_precise_points_in_frame(frame, prompt):
    w, h = frame.size

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {
                    "type": "text",
                    "text": f"Detect all instances of: {prompt}. Return only bounding boxes for objects that exactly match this description."
                }
            ]
        }
    ]
    text = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_v(text=[text], images=[frame], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model_v.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = out[:, inputs.input_ids.shape[1]:]
    txt = processor_v.batch_decode(generated, skip_special_tokens=True)[0]

    bboxes = parse_bboxes_from_text(txt)

    if bboxes:
        points = []
        for b in bboxes:
            bw = abs(b[2] - b[0])
            bh = abs(b[3] - b[1])

            if bw < 5 or bh < 5:
                continue
            if bw > 950 and bh > 950:
                continue

            cx = (b[0] + b[2]) / 2 / 1000 * w
            cy = (b[1] + b[3]) / 2 / 1000 * h

            if 0 <= cx <= w and 0 <= cy <= h:
                points.append((cx, cy))

        if len(points) > 1:
            deduped = [points[0]]
            for pt in points[1:]:
                is_dup = any(pixel_point_distance(pt, ex) < 20 for ex in deduped)
                if not is_dup:
                    deduped.append(pt)
            points = deduped

        if points:
            return points

    messages2 = [
        {"role": "system", "content": [{"type": "text", "text": POINT_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {
                    "type": "text",
                    "text": f"Point to the exact center of each '{prompt}' in this image. Only point to objects that are clearly '{prompt}', nothing else."
                }
            ]
        }
    ]
    text2 = processor_v.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    inputs2 = processor_v(text=[text2], images=[frame], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        out2 = model_v.generate(**inputs2, max_new_tokens=512, do_sample=False)

    generated2 = out2[:, inputs2.input_ids.shape[1]:]
    txt2 = processor_v.batch_decode(generated2, skip_special_tokens=True)[0]

    return parse_precise_points(txt2, w, h)


def track_points_across_frames(pt_state: PointTrackerState, prompt: str):
    total = pt_state.num_frames
    prev_tracks = []
    lost_count = {}

    for f_idx in range(total):
        frame = pt_state.video_frames[f_idx]
        w, h = frame.size

        new_points = detect_precise_points_in_frame(frame, prompt)
        points_f = pt_state.points_by_frame.setdefault(f_idx, [])

        if not prev_tracks:
            for px, py in new_points:
                track_idx = len(pt_state.trails)
                pt_state.trails.append([])
                points_f.append((px, py))
                pt_state.trails[track_idx].append((f_idx, px, py))
                prev_tracks.append((track_idx, (px, py)))
                lost_count[track_idx] = 0
            continue

        if not new_points:
            new_prev = []
            for track_idx, prev_pt in prev_tracks:
                lost_count[track_idx] = lost_count.get(track_idx, 0) + 1
                if lost_count[track_idx] > 5:
                    continue
                points_f.append(prev_pt)
                pt_state.trails[track_idx].append((f_idx, prev_pt[0], prev_pt[1]))
                new_prev.append((track_idx, prev_pt))
            prev_tracks = new_prev
            continue

        diag = (w ** 2 + h ** 2) ** 0.5
        match_threshold = diag * 0.25

        used_new = set()
        matched = {}

        dist_pairs = []
        for pi, (_, prev_pt) in enumerate(prev_tracks):
            for ni, new_pt in enumerate(new_points):
                d = pixel_point_distance(prev_pt, new_pt)
                dist_pairs.append((d, pi, ni))
        dist_pairs.sort()

        for d, pi, ni in dist_pairs:
            if pi in matched or ni in used_new:
                continue
            if d < match_threshold:
                matched[pi] = ni
                used_new.add(ni)

        new_prev = []
        for pi, (track_idx, prev_pt) in enumerate(prev_tracks):
            if pi in matched:
                ni = matched[pi]
                new_pt = new_points[ni]
                points_f.append(new_pt)
                pt_state.trails[track_idx].append((f_idx, new_pt[0], new_pt[1]))
                new_prev.append((track_idx, new_pt))
                lost_count[track_idx] = 0
            else:
                lost_count[track_idx] = lost_count.get(track_idx, 0) + 1
                if lost_count[track_idx] <= 5:
                    points_f.append(prev_pt)
                    pt_state.trails[track_idx].append((f_idx, prev_pt[0], prev_pt[1]))
                    new_prev.append((track_idx, prev_pt))

        for ni, new_pt in enumerate(new_points):
            if ni not in used_new:
                too_close = any(
                    pixel_point_distance(new_pt, prev_pt) < diag * 0.08
                    for _, prev_pt in new_prev
                )
                if not too_close:
                    track_idx = len(pt_state.trails)
                    pt_state.trails.append([])
                    points_f.append(new_pt)
                    pt_state.trails[track_idx].append((f_idx, new_pt[0], new_pt[1]))
                    new_prev.append((track_idx, new_pt))
                    lost_count[track_idx] = 0

        prev_tracks = new_prev


def render_point_tracker_video(pt_state: PointTrackerState, output_fps: int, trail_length: int = 12):
    RED = (255, 40, 40)
    DARK_RED = (180, 0, 0)
    frames_bgr = []

    for i in range(pt_state.num_frames):
        frame = pt_state.video_frames[i].copy()
        draw = ImageDraw.Draw(frame)

        points_f = pt_state.points_by_frame.get(i, [])

        for trail in pt_state.trails:
            trail_pts = [(tx, ty) for fi, tx, ty in trail if fi <= i and fi > i - trail_length]
            if len(trail_pts) >= 2:
                for t_idx in range(len(trail_pts) - 1):
                    alpha_ratio = (t_idx + 1) / len(trail_pts)
                    trail_color = (
                        int(DARK_RED[0] * alpha_ratio),
                        int(DARK_RED[1] * alpha_ratio),
                        int(DARK_RED[2] * alpha_ratio)
                    )
                    thickness = max(1, int(2 * alpha_ratio))
                    x1t, y1t = int(trail_pts[t_idx][0]), int(trail_pts[t_idx][1])
                    x2t, y2t = int(trail_pts[t_idx + 1][0]), int(trail_pts[t_idx + 1][1])
                    draw.line([(x1t, y1t), (x2t, y2t)], fill=trail_color, width=thickness)

        for (px, py) in points_f:
            r_outer = 10
            draw.ellipse(
                (px - r_outer, py - r_outer, px + r_outer, py + r_outer),
                outline="white", width=2
            )
            r = 7
            draw.ellipse(
                (px - r, py - r, px + r, py + r),
                fill=RED, outline=RED
            )
            r_inner = 2
            draw.ellipse(
                (px - r_inner, py - r_inner, px + r_inner, py + r_inner),
                fill=(255, 200, 200)
            )

        frames_bgr.append(np.array(frame)[:, :, ::-1])

        if (i + 1) % 30 == 0:
            gc.collect()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        writer = cv2.VideoWriter(
            tmp.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (frames_bgr[0].shape[1], frames_bgr[0].shape[0])
        )
        for fr in frames_bgr:
            writer.write(fr)
        writer.release()
        return tmp.name

def process_and_render_points(pt_state: PointTrackerState, video, text_prompt: str, output_fps: int, max_seconds: float):
    if video is None:
        return "❌ Please upload a video", None

    if not text_prompt or not text_prompt.strip():
        return "❌ Please enter at least one text prompt", None

    pt_state.reset()

    if isinstance(video, dict):
        path = video.get("name") or video.get("path") or video.get("data")
    else:
        path = video

    frames, info = try_load_video_frames(path)
    if not frames:
        return "❌ Could not load video", None

    if info["fps"] and len(frames) > max_seconds * info["fps"]:
        frames = frames[:int(max_seconds * info["fps"])]

    pt_state.video_frames = frames
    pt_state.video_fps = info["fps"]

    prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]

    status = f"✅ Video loaded: {pt_state.num_frames} frames\n"
    status += f"Output FPS: {output_fps}\n"
    status += f"Max Seconds: {max_seconds}s\n"
    status += f"Processing {len(prompts)} prompt(s) with point tracking...\n\n"

    for p in prompts:
        track_points_across_frames(pt_state, p)
        status += f"• '{p}': tracked\n"

    total_tracked = len(pt_state.trails)
    status += f"\n📍 Total tracked points: {total_tracked}\n"
    status += "\n🎥 Rendering video with red dot tracking..."
    rendered_path = render_point_tracker_video(pt_state, output_fps)
    status += "\n\n✅ Done! Play the video below."

    return status, rendered_path


with gr.Blocks() as demo:
    gr.Markdown("# Points Tracker")

    gr.Markdown(
        """
        Upload a video, enter one or more prompts separated by commas,
        and generate a tracked video with red dots and motion trails.
        """
    )

    pt_state = gr.State(PointTrackerState())

    with gr.Row():
        with gr.Column():
            pt_video_in = gr.Video(label="Upload Video", sources=["upload", "webcam"], height=400)
            pt_prompt_in = gr.Textbox(
                label="Text Prompts (comma separated)",
                placeholder="person, ball, car, face, hand",
                lines=3
            )
            pt_fps_slider = gr.Slider(
                label="Output Video FPS",
                minimum=1,
                maximum=60,
                value=25,
                step=1
            )
            pt_max_seconds_slider = gr.Slider(
                label="Max Video Seconds",
                minimum=1,
                maximum=MAX_SECONDS_LIMIT,
                value=DEFAULT_MAX_SECONDS,
                step=1
            )
            pt_process_btn = gr.Button("Apply Point Tracking & Render Video", variant="primary")

        with gr.Column():
            pt_status_out = gr.Textbox(label="Output Status", lines=10)
            pt_rendered_out = gr.Video(label="Rendered Video with Point Tracking", height=400)

    pt_process_btn.click(
        fn=process_and_render_points,
        inputs=[pt_state, pt_video_in, pt_prompt_in, pt_fps_slider, pt_max_seconds_slider],
        outputs=[pt_status_out, pt_rendered_out],
        show_progress=True
    )

if __name__ == "__main__":
    demo.queue().launch(show_error=True, ssr_mode=False)
