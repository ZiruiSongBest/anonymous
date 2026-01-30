from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from rembg import remove
import argparse


def build_affordance_prompt(
    query: str,
    channel_key: str = "c0",
    geometry: str = "",
    include_geometry: bool = True,
    include_image_token: bool = True,
) -> str:
    lines = []
    if include_image_token:
        lines.append("<image>")
    lines += [
        "Affordance_queries:",
        f"{channel_key}: {query}",
        "",
        "Affordance_voxels (voxel grid=32, indices 0-32767, merge runs):",
    ]
    if include_geometry:
        if geometry:
            lines.append(f"geometry: {geometry}")
        else:
            lines.append("geometry: <indices>")
    lines.append(f"{channel_key}: <indices>")
    return "\n".join(lines)


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def ensure_chat_template(processor) -> None:
    template = (
        "{% set image_count = namespace(value=0) %}"
        "{% set video_count = namespace(value=0) %}"
        "{% for message in messages %}"
        "{% if loop.first and message['role'] != 'system' %}"
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "{% endif %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}<|im_end|>\n"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
        "{% set image_count.value = image_count.value + 1 %}"
        "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "{% elif content['type'] == 'video' or 'video' in content %}"
        "{% set video_count.value = video_count.value + 1 %}"
        "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
        "<|vision_start|><|video_pad|><|vision_end|>"
        "{% elif 'text' in content %}"
        "{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    current = getattr(processor, "chat_template", None)
    if current is None or "vision_start" not in current:
        processor.chat_template = template
    if hasattr(processor, "tokenizer"):
        tok_current = getattr(processor.tokenizer, "chat_template", None)
        if tok_current is None or "vision_start" not in tok_current:
            processor.tokenizer.chat_template = template


def generate_text(model, messages, max_new_tokens=4096, max_length=None):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    gen_kwargs = {
        "do_sample": False,
        "temperature": 0,
    }
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
    else:
        gen_kwargs["max_length"] = max_length or 32768

    generated_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove_bg", type=bool, default=False)
    parser.add_argument("--ckpt", type=str, default="./model/vlm")
    parser.add_argument("--processor_path", type=str, default="./model/vlm")
    parser.add_argument("--image_path", type=str, default="./example")
    parser.add_argument("--query", type=str, default="grasp")
    parser.add_argument("--channel_key", type=str, default="c0")
    parser.add_argument("--geometry", type=str, default="")
    parser.add_argument("--geometry_file", type=str, default="")
    parser.add_argument("--no_geometry", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--save_output", type=str, default="")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        device_map = {"": "cpu"}
        max_mem = {"cpu": "64GiB"}
    elif num_gpus == 1:
        device_map = {"": 0}
        max_mem = {0: "120GiB"}
    else:
        device_map = "auto"
        max_mem = {i: "120GiB" for i in range(num_gpus)}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
        max_memory=max_mem,
    )

    min_pixels = 65536
    max_pixels = 262144

    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.size["shortest_edge"] = min_pixels
    processor.image_processor.size["longest_edge"] = max_pixels
    ensure_chat_template(processor)

    if not args.image_path:
        raise SystemExit("--image_path is required for affordance inference.")
    geometry_text = args.geometry
    if args.geometry_file:
        geometry_text = load_text_file(args.geometry_file)

    input_image = Image.open(args.image_path)
    im_resized = input_image.resize((512, 512), Image.LANCZOS)
    if args.remove_bg:
        im_resized = remove(im_resized)

    prompt = build_affordance_prompt(
        query=args.query,
        channel_key=args.channel_key,
        geometry=geometry_text,
        include_geometry=not args.no_geometry,
        include_image_token=False,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": im_resized.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    output_text = generate_text(
        model,
        messages,
        max_new_tokens=args.max_new_tokens,
    )
    if args.save_output:
        with open(args.save_output, "w", encoding="utf-8") as f:
            f.write(output_text)
    else:
        print(output_text)
