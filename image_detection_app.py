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


def annotate_detection_image(image: Image.Image, result: dict):
    if not isinstance(image, Image.Image) or not isinstance(result, dict):
        return image

    image = image.convert("RGB")
    ow, oh = image.size

    if "objects" not in result or not result["objects"]:
        return image

    boxes, labels = [], []
    for obj in result["objects"]:
        boxes.append([
            obj.get("x_min", 0.0) * ow,
            obj.get("y_min", 0.0) * oh,
            obj.get("x_max", 0.0) * ow,
            obj.get("y_max", 0.0) * oh,
        ])
        labels.append(obj.get("label", "object"))

    if not boxes:
        return image

    scene = np.array(image.copy())
    h, w = scene.shape[:2]
    masks = np.zeros((len(boxes), h, w), dtype=bool)

    for i, box in enumerate(boxes):
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
        masks[i, y1:y2, x1:x2] = True

    dets = sv.Detections(xyxy=np.array(boxes), mask=masks)
    if len(dets) == 0:
        return image

    scene = sv.MaskAnnotator(
        color=BRIGHT_YELLOW,
        opacity=0.18,
        color_lookup=sv.ColorLookup.INDEX
    ).annotate(scene=scene, detections=dets)

    scene = sv.BoxAnnotator(
        color=BRIGHT_YELLOW,
        thickness=2,
        color_lookup=sv.ColorLookup.INDEX
    ).annotate(scene=scene, detections=dets)

    scene = sv.LabelAnnotator(
        color=BRIGHT_YELLOW,
        text_color=BLACK,
        text_scale=0.5,
        text_thickness=1,
        text_padding=6,
        color_lookup=sv.ColorLookup.INDEX,
    ).annotate(scene=scene, detections=dets, labels=labels)

    return Image.fromarray(scene)


def process_image_detection(image, prompt):
    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a detection prompt.")

    image = image.convert("RGB")
    image.thumbnail((512, 512))

    full_prompt = f"Provide bounding box coordinates for {prompt}. Report in JSON format."

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": full_prompt}
        ]
    }]

    text = qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = qwen_processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(qwen_model.device)

    streamer = TextIteratorStreamer(
        qwen_processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120
    )

    thread = Thread(
        target=qwen_model.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            use_cache=True,
            temperature=1.5,
            min_p=0.1
        )
    )
    thread.start()

    full_text = ""
    for tok in streamer:
        full_text += tok
        yield image, full_text

    thread.join()

    parsed = safe_parse_json(full_text)
    result = {"objects": []}

    if isinstance(parsed, list):
        for item in parsed:
            if "bbox_2d" in item and len(item["bbox_2d"]) == 4:
                xmin, ymin, xmax, ymax = item["bbox_2d"]
                result["objects"].append({
                    "label": item.get("label", "object"),
                    "x_min": xmin / 1000.0,
                    "y_min": ymin / 1000.0,
                    "x_max": xmax / 1000.0,
                    "y_max": ymax / 1000.0,
                })

    annotated = annotate_detection_image(image.copy(), result)
    yield annotated, json.dumps(result, indent=2)


with gr.Blocks() as demo:
    gr.Markdown("# Image Detection")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(
                label="Detection Prompt",
                placeholder="e.g. all cars, all persons, the cat"
            )
            run_btn = gr.Button("Detect")

        with gr.Column():
            output_image = gr.Image(label="Annotated Image")
            output_text = gr.Textbox(label="Detection Output", lines=12)

    run_btn.click(
        fn=process_image_detection,
        inputs=[image_input, prompt_input],
        outputs=[output_image, output_text],
    )


if __name__ == "__main__":
    demo.launch(show_error=True, ssr_mode=False)