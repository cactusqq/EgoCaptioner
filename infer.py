#!/usr/bin/env python3
"""Caption one egocentric video with the merged EgoCaptioner model."""

import argparse
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
import sys


PROMPTS = Path(__file__).resolve().parent / "prompts"
VIDEO_MAX_PIXELS = 150528


def build_prompt(task, duration, event=None):
    if task not in ("full", "overview", "event_steps"):
        raise ValueError(f"Unknown task: {task}")
    if task == "event_steps" and not (event and event.strip()):
        raise ValueError("event_steps requires --event with an event description")
    value = Decimal(str(duration))
    if not value.is_finite() or value <= 0:
        raise ValueError("Video duration must be finite and positive")
    # Match training: show seconds rounded down to one decimal place.
    shown = (value + Decimal("0.000001")).quantize(Decimal("0.1"), rounding=ROUND_FLOOR)
    if shown <= 0:
        raise ValueError("The video must be at least 0.1 seconds long")
    template = (PROMPTS / f"{task}.txt").read_text(encoding="utf-8").strip()
    return template.format(duration=f"{shown:.1f}", event=event)


def video_duration(path):
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path}")
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return container.duration / av.time_base
    raise ValueError(f"Cannot determine video duration: {path}")


def prepare_video(processor, video, task, event=None, fps=4.0, max_frames=240):
    """Share prompt construction, sampled frames, and timestamps across backends."""
    from qwen_vl_utils import process_vision_info

    duration = video_duration(video)
    prompt = build_prompt(task, duration, event)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": str(Path(video).resolve()), "fps": fps,
         "max_frames": max_frames, "min_pixels": 65536, "max_pixels": VIDEO_MAX_PIXELS},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    _, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True,
        return_video_metadata=True)
    frames, metadata = videos[0]
    return text, videos, video_kwargs, {
        "duration_seconds": duration, "sampled_frames": len(frames),
        "video_backend": metadata.get("video_backend", "unknown"), "prompt": prompt}


def prepare_inputs(processor, video, task, event=None, fps=4.0, max_frames=240):
    text, videos, video_kwargs, metadata = prepare_video(
        processor, video, task, event, fps, max_frames)
    frames, video_metadata = zip(*videos)
    inputs = processor(
        text=[text], videos=list(frames), video_metadata=list(video_metadata),
        do_resize=False, return_tensors="pt", **video_kwargs)
    return inputs, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Merged model directory or Hugging Face model ID")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--task", choices=("full", "overview", "event_steps"), default="full")
    parser.add_argument("--event", help="Event to locate and describe; required for event_steps")
    parser.add_argument("--output", type=Path, help="Also save the caption as UTF-8 text")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0, help="Set 0 for greedy decoding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa"),
                        default="flash_attention_2", help="Attention backend (default: flash_attention_2)")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    if args.task == "event_steps" and not (args.event and args.event.strip()):
        parser.error("--event is required for event_steps")
    if not 0 < args.fps < float("inf"):
        parser.error("--fps must be finite and positive")
    if args.max_frames < 4 or args.max_frames % 2:
        parser.error("--max-frames must be an even number of at least 4")
    if args.max_new_tokens < 1 or not 0 <= args.temperature < float("inf"):
        parser.error("Use positive --max-new-tokens and finite, nonnegative --temperature")
    if (Path(args.model) / "lora_adapter").is_dir():
        parser.error("This is an unmerged training export. Use the merged release model.")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

    if args.attn_implementation == "flash_attention_2":
        if not torch.cuda.is_available():
            parser.error("FlashAttention 2 requires a CUDA GPU. For SDPA, use --attn-implementation sdpa.")
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

        if not is_fast_path_available:
            parser.error("The Qwen3.5 linear-attention kernels are unavailable. Install the dependencies "
                         "in requirements.txt and requirements-kernels.txt as described in the README.")

    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model)
    inputs, metadata = prepare_inputs(
        processor, args.video, args.task, args.event, args.fps, args.max_frames)
    print(f"Video: {metadata['duration_seconds']:.2f}s, "
          f"sampled frames: {metadata['sampled_frames']}, "
          f"decoder: {metadata['video_backend']}", file=sys.stderr)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation=args.attn_implementation).eval()
    inputs = inputs.to(model.device)
    generation = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0}
    if args.temperature > 0:
        generation.update(temperature=args.temperature, top_p=0.95, top_k=20)
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation)
    new_tokens = outputs[:, inputs.input_ids.shape[1]:]
    caption = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if "</think>" in caption:
        caption = caption.partition("</think>")[2]
    caption = caption.strip()
    if new_tokens.shape[1] >= args.max_new_tokens:
        print("The generation reached --max-new-tokens; the caption may be incomplete.", file=sys.stderr)
    print(caption)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(caption + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
