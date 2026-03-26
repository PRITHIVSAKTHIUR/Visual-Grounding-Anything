# Install the required dependencies before running this script:
# pip install torch torchvision
# pip install gradio==6.9.0
# pip install transformers==5.3.0
# pip install supervision==0.27.0.post2

import gradio as gr
import torch
import numpy as np
import supervision as sv
import json
import ast
import re
from PIL import Image
from threading import Thread
from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

MODEL_NAME = "prithivMLmods/Polaris-VGA-4B-Post1.0e"

BRIGHT_YELLOW = sv.Color(r=255, g=230, b=0)
DARK_OUTLINE = sv.Color(r=40, g=40, b=40)
BLACK = sv.Color(r=0, g=0, b=0)

print(f"Loading model: {MODEL_NAME} ...")
qwen_model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map=DEVICE,
).eval()

qwen_processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")


def safe_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def normalize_1000_coord(v):
    try:
        v = float(v)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v / 1000.0))


def parse_pointer_response(raw_text: str):
    parsed = safe_parse_json(raw_text)
    result = {"points": []}

    if not isinstance(parsed, list):
        return result

    for item in parsed:
        if not isinstance(item, dict):
            continue

        label = item.get("label", "")

        # Preferred format: point_2d
        if "point_2d" in item and isinstance(item["point_2d"], (list, tuple)) and len(item["point_2d"]) == 2:
            x, y = item["point_2d"]
            result["points"].append({
                "label": label,
                "x": normalize_1000_coord(x),
                "y": normalize_1000_coord(y),
            })
            continue

        # Fallback: bbox_2d -> convert bbox center to point
        if "bbox_2d" in item and isinstance(item["bbox_2d"], (list, tuple)) and len(item["bbox_2d"]) == 4:
            x1, y1, x2, y2 = item["bbox_2d"]
            cx = (float(x1) + float(x2)) / 2.0
            cy = (float(y1) + float(y2)) / 2.0
            result["points"].append({
                "label": label,
                "x": normalize_1000_coord(cx),
                "y": normalize_1000_coord(cy),
            })

    return result


def annotate_point_image(image: Image.Image, result: dict):
    if not isinstance(image, Image.Image) or not isinstance(result, dict):
        return image

    image = image.convert("RGB")
    ow, oh = image.size

    if "points" not in result or not result["points"]:
        return image

    pts = []
    valid_points = []

    for p in result["points"]:
        if "x" not in p or "y" not in p:
            continue
        px = int(float(p["x"]) * ow)
        py = int(float(p["y"]) * oh)
        px = max(0, min(ow - 1, px))
        py = max(0, min(oh - 1, py))
        pts.append([px, py])
        valid_points.append(p)

    if not pts:
        return image

    kp = sv.KeyPoints(xy=np.array(pts, dtype=np.int32).reshape(1, -1, 2))
    scene = np.array(image.copy())

    scene = sv.VertexAnnotator(radius=10, color=DARK_OUTLINE).annotate(
        scene=scene, key_points=kp
    )
    scene = sv.VertexAnnotator(radius=6, color=BRIGHT_YELLOW).annotate(
        scene=scene, key_points=kp
    )

    labels = [p.get("label", "") for p in valid_points]
    if any(labels):
        tb, vl = [], []
        for i, p in enumerate(valid_points):
            if labels[i]:
                cx, cy = pts[i]
                tb.append([cx - 2, cy - 2, cx + 2, cy + 2])
                vl.append(labels[i])

        if tb:
            scene = sv.LabelAnnotator(
                color=BRIGHT_YELLOW,
                text_color=BLACK,
                text_scale=0.5,
                text_thickness=1,
                text_padding=5,
                text_position=sv.Position.TOP_CENTER,
                color_lookup=sv.ColorLookup.INDEX,
            ).annotate(
                scene=scene,
                detections=sv.Detections(xyxy=np.array(tb)),
                labels=vl,
            )

    return Image.fromarray(scene)


def process_image_pointer(image, prompt):
    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a point prompt.")

    original_image = image.convert("RGB")
    model_image = original_image.copy()
    model_image.thumbnail((512, 512))

    full_prompt = (
        f"Provide 2d point coordinates for {prompt}. "
        f"Report in JSON format as a list like "
        f'[{{"point_2d": [x, y], "label": "{prompt}"}}]. '
        f"If you cannot provide points, do not return bounding boxes."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": model_image},
            {"type": "text", "text": full_prompt},
        ],
    }]

    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = qwen_processor(
        text=[text],
        images=[model_image],
        return_tensors="pt",
        padding=True,
    ).to(qwen_model.device)

    streamer = TextIteratorStreamer(
        qwen_processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120,
    )

    thread = Thread(
        target=qwen_model.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        ),
    )
    thread.start()

    full_text = ""
    for tok in streamer:
        full_text += tok
        yield original_image, full_text

    thread.join()

    result = parse_pointer_response(full_text)
    annotated = annotate_point_image(original_image.copy(), result)

    yield annotated, json.dumps(result, indent=2)


with gr.Blocks() as demo:
    gr.Markdown("# Image Pointer")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(
                label="Point Prompt",
                placeholder="e.g. left eye, car headlight, person hand"
            )
            button = gr.Button("Run Pointing")

        with gr.Column():
            output_image = gr.Image(label="Pointed Image")
            output_text = gr.Textbox(label="Point Output", lines=12)

    button.click(
        fn=process_image_pointer,
        inputs=[image_input, prompt_input],
        outputs=[output_image, output_text],
    )


if __name__ == "__main__":
    demo.launch(show_error=True, ssr_mode=False)