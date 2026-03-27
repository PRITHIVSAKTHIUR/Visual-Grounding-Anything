# Multi-Stage Heuristic Matching.
# Pass 1: Greedy IoU Matching (The Primary Tracker)
# Pass 2: Euclidean Distance Fallback (The Occlusion/Jitter Handler)

# Install the required dependencies before running this script:
# pip install torch torchvision
# pip install gradio==6.9.0
# pip install transformers==5.3.0
# pip install opencv-python==4.13.0.92

import colorsys
import gc
import tempfile
import re
import json
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
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


def bbox_to_mask(bbox_scaled, width, height):
    mask = np.zeros((height, width), dtype=np.float32)
    x1 = max(0, min(int(bbox_scaled[0] / 1000 * width), width - 1))
    y1 = max(0, min(int(bbox_scaled[1] / 1000 * height), height - 1))
    x2 = max(0, min(int(bbox_scaled[2] / 1000 * width), width - 1))
    y2 = max(0, min(int(bbox_scaled[3] / 1000 * height), height - 1))
    mask[y1:y2, x1:x2] = 1.0
    return mask


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / union if union > 0 else 0.0


def bbox_center_distance(b1, b2):
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def overlay_masks_on_frame(frame, masks, colors_map, alpha=0.45):
    base = np.array(frame).astype(np.float32) / 255
    overlay = base.copy()
    for oid, mask in masks.items():
        if mask is None:
            continue
        color = np.array(colors_map.get(oid, (255, 0, 0)), dtype=np.float32) / 255
        m = np.clip(mask, 0, 1)[..., None]
        overlay = (1 - alpha * m) * overlay + (alpha * m) * color
    return Image.fromarray(np.clip(overlay * 255, 0, 255).astype(np.uint8))


def pastel_color_for_prompt(prompt: str):
    hue = (sum(ord(c) for c in prompt) * 2654435761 % 360) / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


class AppState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.video_frames = []
        self.video_fps = None
        self.masks_by_frame = {}
        self.bboxes_by_frame = {}
        self.color_by_obj = {}
        self.color_by_prompt = {}
        self.text_prompts_by_frame_obj = {}
        self.prompts = {}
        self.next_obj_id = 1

    @property
    def num_frames(self):
        return len(self.video_frames)


def detect_objects_in_frame(frame, prompt):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": f"Detect all instances of: {prompt}"}
            ]
        }
    ]
    text = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_v(text=[text], images=[frame], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model_v.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = out[:, inputs.input_ids.shape[1]:]
    txt = processor_v.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_bboxes_from_text(txt)


def track_prompt_across_frames(state: AppState, prompt: str):
    total = state.num_frames

    if prompt in state.prompts:
        for oid in state.prompts[prompt]:
            for f in range(total):
                state.masks_by_frame[f].pop(oid, None)
                state.bboxes_by_frame[f].pop(oid, None)
                state.text_prompts_by_frame_obj[f].pop(oid, None)
        del state.prompts[prompt]

    prev_tracks = []

    for f_idx in range(total):
        frame = state.video_frames[f_idx]
        w, h = frame.size
        new_bboxes = detect_objects_in_frame(frame, prompt)

        masks_f = state.masks_by_frame.setdefault(f_idx, {})
        bboxes_f = state.bboxes_by_frame.setdefault(f_idx, {})
        texts_f = state.text_prompts_by_frame_obj.setdefault(f_idx, {})

        if not prev_tracks:
            for bbox in new_bboxes:
                oid = state.next_obj_id
                state.next_obj_id += 1
                if prompt not in state.color_by_prompt:
                    state.color_by_prompt[prompt] = pastel_color_for_prompt(prompt)
                state.color_by_obj[oid] = state.color_by_prompt[prompt]
                masks_f[oid] = bbox_to_mask(bbox, w, h)
                bboxes_f[oid] = bbox
                texts_f[oid] = prompt
                state.prompts.setdefault(prompt, []).append(oid)
                prev_tracks.append((oid, bbox))
            continue

        used = set()
        matched = {}

        scores = [
            (bbox_iou(pbbox, nbbox), pi, ni)
            for pi, (_, pbbox) in enumerate(prev_tracks)
            for ni, nbbox in enumerate(new_bboxes)
        ]
        scores.sort(reverse=True)

        for score, pi, ni in scores:
            if pi in matched or ni in used or score <= 0.05:
                continue
            matched[pi] = ni
            used.add(ni)

        for pi, (_, pbbox) in enumerate(prev_tracks):
            if pi in matched:
                continue
            best = min(
                (
                    (bbox_center_distance(pbbox, nbbox), ni)
                    for ni, nbbox in enumerate(new_bboxes) if ni not in used
                ),
                default=(float('inf'), -1)
            )
            if best[0] < 300:
                matched[pi] = best[1]
                used.add(best[1])

        new_prev = []
        for pi, (oid, _) in enumerate(prev_tracks):
            if pi in matched:
                nbbox = new_bboxes[matched[pi]]
                masks_f[oid] = bbox_to_mask(nbbox, w, h)
                bboxes_f[oid] = nbbox
                texts_f[oid] = prompt
                new_prev.append((oid, nbbox))

        for ni, nbbox in enumerate(new_bboxes):
            if ni not in used:
                oid = state.next_obj_id
                state.next_obj_id += 1
                if prompt not in state.color_by_prompt:
                    state.color_by_prompt[prompt] = pastel_color_for_prompt(prompt)
                state.color_by_obj[oid] = state.color_by_prompt[prompt]
                masks_f[oid] = bbox_to_mask(nbbox, w, h)
                bboxes_f[oid] = nbbox
                texts_f[oid] = prompt
                state.prompts.setdefault(prompt, []).append(oid)
                new_prev.append((oid, nbbox))

        prev_tracks = new_prev


def render_full_video(state: AppState, output_fps: int):
    frames_bgr = []

    for i in range(state.num_frames):
        frame = state.video_frames[i].copy()
        masks = state.masks_by_frame.get(i, {})
        if masks:
            frame = overlay_masks_on_frame(frame, masks, state.color_by_obj)

        bboxes = state.bboxes_by_frame.get(i, {})
        if bboxes:
            draw = ImageDraw.Draw(frame)
            w, h = frame.size
            for oid, bbox in bboxes.items():
                color = state.color_by_obj.get(oid, (255, 255, 255))
                x1 = int(bbox[0] / 1000 * w)
                y1 = int(bbox[1] / 1000 * h)
                x2 = int(bbox[2] / 1000 * w)
                y2 = int(bbox[3] / 1000 * h)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=4)

                prompt = state.text_prompts_by_frame_obj.get(i, {}).get(oid, "")
                if prompt:
                    label = f"{prompt} - ID{oid}"
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                    except OSError:
                        font = ImageFont.load_default()

                    tb = draw.textbbox((x1, max(0, y1 - 30)), label, font=font)
                    draw.rectangle(tb, fill=color)
                    draw.text((x1 + 4, max(0, y1 - 27)), label, fill="white", font=font)

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

def process_and_render(state: AppState, video, text_prompt: str, output_fps: int, max_seconds: float):
    if video is None:
        return "❌ Please upload a video", None

    if not text_prompt or not text_prompt.strip():
        return "❌ Please enter at least one text prompt", None

    state.reset()

    if isinstance(video, dict):
        path = video.get("name") or video.get("path") or video.get("data")
    else:
        path = video

    frames, info = try_load_video_frames(path)
    if not frames:
        return "❌ Could not load video", None

    if info["fps"] and len(frames) > max_seconds * info["fps"]:
        frames = frames[:int(max_seconds * info["fps"])]

    state.video_frames = frames
    state.video_fps = info["fps"]

    prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]

    status = f"✅ Video loaded: {state.num_frames} frames\n"
    status += f"Output FPS: {output_fps}\n"
    status += f"Max Seconds: {max_seconds}s\n"
    status += f"Processing {len(prompts)} prompt(s) across all frames...\n\n"

    for p in prompts:
        track_prompt_across_frames(state, p)
        count = len(state.prompts.get(p, []))
        status += f"• '{p}': {count} object(s) tracked\n"

    status += "\n🎥 Rendering final video with overlays..."
    rendered_path = render_full_video(state, output_fps)
    status += "\n\n✅ Done! Play the video below."

    return status, rendered_path


with gr.Blocks() as demo:
    gr.Markdown("# Object Tracking")

    gr.Markdown(
        """
        Upload a video, enter one or more object prompts separated by commas,
        and generate a tracked video with masks, boxes, and IDs.
        """
    )

    state = gr.State(AppState())

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Upload Video", sources=["upload", "webcam"], height=400)
            prompt_in = gr.Textbox(
                label="Text Prompts (comma separated)",
                placeholder="person, red car, dog, laptop, traffic light",
                lines=3
            )
            fps_slider = gr.Slider(
                label="Output Video FPS",
                minimum=1,
                maximum=60,
                value=25,
                step=1
            )
            max_seconds_slider = gr.Slider(
                label="Max Video Seconds",
                minimum=1,
                maximum=MAX_SECONDS_LIMIT,
                value=DEFAULT_MAX_SECONDS,
                step=1
            )
            process_btn = gr.Button("Apply Detection and Render Full Video", variant="primary")

        with gr.Column():
            status_out = gr.Textbox(label="Output Status", lines=10)
            rendered_out = gr.Video(label="Rendered Video with Object Tracking", height=400)

    process_btn.click(
        fn=process_and_render,
        inputs=[state, video_in, prompt_in, fps_slider, max_seconds_slider],
        outputs=[status_out, rendered_out],
        show_progress=True
    )

if __name__ == "__main__":
    demo.queue().launch(show_error=True, ssr_mode=False)