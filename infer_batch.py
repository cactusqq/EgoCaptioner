#!/usr/bin/env python3
"""Caption videos from JSONL with vLLM continuous batching."""

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import sys
import time

from infer import PROMPTS, VIDEO_MAX_PIXELS, prepare_video


def invalid_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def read_rows(path, completed):
    """Read metadata lazily; paths are relative to the input JSONL."""
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line, text in enumerate(source, 1):
            if not text.strip():
                continue
            row = {"line": line, "id": f"line-{line}"}
            try:
                value = json.loads(text, parse_constant=invalid_constant)
                if not isinstance(value, dict):
                    raise ValueError("Each line must be a JSON object")
                row.update({k: value[k] for k in ("id", "video", "task", "event") if k in value})
                if not isinstance(row["id"], str) or not row["id"].strip():
                    raise ValueError("id must be a nonempty string")
                if row["id"] in seen:
                    raise ValueError(f"Duplicate id: {row['id']}")
                seen.add(row["id"])
                row.setdefault("task", "full")
                if row["task"] not in ("full", "overview", "event_steps"):
                    raise ValueError("task must be full, overview, or event_steps")
                event = row.get("event")
                if event is not None and not isinstance(event, str):
                    raise ValueError("event must be a string")
                if row["task"] == "event_steps" and not (event and event.strip()):
                    raise ValueError("event_steps requires a nonempty event description")
                if not isinstance(row.get("video"), str) or not row["video"].strip():
                    raise ValueError("video must be a local video path")
                video = Path(row["video"]).expanduser()
                video = (path.parent / video).resolve()
                if not video.is_file():
                    raise ValueError(f"Video does not exist: {video}")
                row["video"] = str(video)
            except (ValueError, TypeError, OSError) as exc:
                row.update(status="error", error=str(exc))
            if line not in completed:
                yield row


def run_config(args):
    with args.input.open("rb") as source:
        input_hash = hashlib.file_digest(source, "sha256").hexdigest()
    model = Path(args.model).expanduser()
    return {"input_sha256": input_hash, "input": str(args.input.resolve()),
            "model": str(model.resolve()) if model.exists() else args.model,
            "backend": "vllm", "enable_thinking": False,
            "top_p": 0.95 if args.temperature > 0 else 1.0,
            "top_k": 20 if args.temperature > 0 else -1,
            **{k: getattr(args, k) for k in ("fps", "max_frames", "max_new_tokens",
                                           "temperature", "seed", "max_model_len")},
            "prompts": {p.name: p.read_text(encoding="utf-8") for p in sorted(PROMPTS.glob("*.txt"))}}


@contextmanager
def open_output(path, config, resume):
    """Lock one append-only result file and recover an interrupted final write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config_path = path.with_name(path.name + ".config.json")
    with path.open("r+b" if resume else "x+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Resume requires the same input, model, prompts, and generation settings")
        else:
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        completed = set()
        while raw := stream.readline():
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                if raw.endswith(b"\n"):
                    raise ValueError("Invalid JSON in the existing output; use a new output path")
                # Only an incomplete final record can be discarded on resume.
                stream.seek(-len(raw), os.SEEK_CUR)
                stream.truncate()
                print("Discarded an incomplete final output record.", file=sys.stderr)
                break
            if result.get("status") == "ok":
                completed.add(result["line"])
            if not raw.endswith(b"\n"):
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                break
        stream.seek(0, os.SEEK_END)

        def save(result):
            stream.write((json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())

        yield completed, save


def fatal_error(exc):
    """Do not treat resource exhaustion or a dead engine as a bad video."""
    while exc is not None:
        if isinstance(exc, MemoryError) or type(exc).__name__ in ("OutOfMemoryError", "EngineDeadError"):
            return True
        if any(text in str(exc).lower() for text in ("cuda out of memory", "illegal memory access")):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def configure_parallelism(args, visible_gpus):
    """Resolve local replica and worker counts before creating the engine."""
    if visible_gpus == 0:
        raise ValueError("Batch inference requires a visible CUDA GPU")
    tp = args.tensor_parallel_size
    if args.data_parallel_size is None:
        args.data_parallel_size = visible_gpus // tp
    dp = args.data_parallel_size
    if dp == 0 or dp * tp > visible_gpus:
        raise ValueError(f"Requested topology does not fit {visible_gpus} visible GPUs "
                         f"(TP={tp}, DP={dp})")
    if args.workers is None:
        args.workers = len(os.sched_getaffinity(0))
    if args.concurrency is None:
        args.concurrency = 8 * dp


async def run_pipeline(rows, prepare, generate, save, workers, concurrency):
    """Bound decoding, ready inputs, in-flight generation, and output separately."""
    inputs = asyncio.Queue(maxsize=2 * workers)
    ready = asyncio.Queue(maxsize=min(workers, concurrency))
    results = asyncio.Queue(maxsize=concurrency)
    counts = Counter()
    started = time.monotonic()

    async def produce():
        for row in rows:
            await inputs.put(row)
        for _ in range(workers):
            await inputs.put(None)

    async def decode():
        while (row := await inputs.get()) is not None:
            if row.get("status") == "error":
                await results.put(row)
                continue
            try:
                prepared, metadata = await prepare(row)
            except Exception as exc:
                if fatal_error(exc):
                    raise
                await results.put({**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                continue
            await ready.put((row, prepared, metadata))
            del prepared, metadata

    async def decode_all():
        async with asyncio.TaskGroup() as group:
            group.create_task(produce())
            for _ in range(workers):
                group.create_task(decode())
        for _ in range(concurrency):
            await ready.put(None)

    async def infer():
        while (item := await ready.get()) is not None:
            row, prepared, metadata = item
            del item
            try:
                result = {**row, **metadata, **await generate(prepared, row), "status": "ok"}
            except Exception as exc:
                if fatal_error(exc):
                    raise
                result = {**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            del prepared
            await results.put(result)

    async def infer_all():
        async with asyncio.TaskGroup() as group:
            for _ in range(concurrency):
                group.create_task(infer())
        await results.put(None)

    async def write():
        while (result := await results.get()) is not None:
            await asyncio.to_thread(save, result)
            counts[result["status"]] += 1
            elapsed = time.monotonic() - started
            detail = f" | {result['error']}" if result["status"] == "error" else ""
            print(f"Completed {sum(counts.values())} | ok={counts['ok']} error={counts['error']} | "
                  f"{elapsed:.1f}s | {result['id']}{detail}", file=sys.stderr, flush=True)

    async with asyncio.TaskGroup() as group:
        group.create_task(decode_all())
        group.create_task(infer_all())
        group.create_task(write())
    return counts


async def run(args, rows, save):
    from transformers import AutoProcessor
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    processor = AutoProcessor.from_pretrained(args.model)
    # vLLM 0.19.1 uses max_num_batched_tokens as the encoder cache floor.
    # Each visual token covers a merged spatial patch across a temporal patch.
    vp = processor.video_processor
    temporal_groups = (args.max_frames + vp.temporal_patch_size - 1) // vp.temporal_patch_size
    visual_tokens = temporal_groups * (VIDEO_MAX_PIXELS // (vp.patch_size * vp.merge_size) ** 2)
    seqs_per_replica = (args.concurrency + args.data_parallel_size - 1) // args.data_parallel_size
    token_budget = min(args.max_model_len, max(2048, visual_tokens, seqs_per_replica))
    print(f"Per-replica scheduler token budget: {token_budget}", file=sys.stderr, flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=args.model, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
        max_num_seqs=seqs_per_replica, max_num_batched_tokens=token_budget,
        seed=args.seed, generation_config="vllm",
        renderer_num_workers=args.workers, mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 0, "video": 1},
        mm_processor_kwargs={"min_pixels": 65536, "max_pixels": VIDEO_MAX_PIXELS,
                             "fps": args.fps, "max_frames": args.max_frames}))
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            async def prepare(row):
                text, videos, kwargs, metadata = await asyncio.get_running_loop().run_in_executor(
                    pool, prepare_video, processor, row["video"], row["task"], row.get("event"),
                    args.fps, args.max_frames)
                # Preserve frame indices and original FPS for timestamp expansion.
                prompt = {"prompt": text, "multi_modal_data": {"video": videos},
                          "mm_processor_kwargs": {**kwargs, "do_resize": False}}
                rendered = await engine.renderer.render_cmpl_async([prompt])
                metadata.pop("prompt")
                return rendered[0], metadata

            async def generate(prepared, row):
                params = SamplingParams(
                    max_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=0.95 if args.temperature > 0 else 1.0,
                    top_k=20 if args.temperature > 0 else -1,
                    seed=args.seed, output_kind=RequestOutputKind.FINAL_ONLY)
                final = None
                async for output in engine.generate(prepared, params, request_id=f"line-{row['line']}"):
                    final = output
                if final is None or not final.outputs:
                    raise ValueError("The model returned no output")
                output = final.outputs[0]
                caption = output.text.rsplit("</think>", 1)[-1].strip()
                if not caption:
                    raise ValueError("The model returned an empty caption")
                return {"caption": caption, "finish_reason": output.finish_reason,
                        "generated_tokens": len(output.token_ids)}

            return await run_pipeline(rows, prepare, generate, save, args.workers, args.concurrency)
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Merged model directory or Hugging Face model ID")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL; one task per line")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL, saved as tasks finish")
    parser.add_argument("--resume", action="store_true", help="Skip successes in an existing output and retry errors")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, help="Video preprocessing workers (default: CPU affinity count)")
    parser.add_argument("--concurrency", type=int, help="Total concurrent generation requests (default: 8 per replica)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="GPUs used by one model engine")
    parser.add_argument("--data-parallel-size", type=int, help="Model replicas (default: all complete groups of visible GPUs)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=65536, help="Total context budget (input + output)")
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.input.is_file() or args.input == args.output:
        parser.error("--input must exist and differ from --output")
    if args.output.exists() != args.resume:
        parser.error("Use a new --output path, or --resume with an existing output")
    if not 0 < args.fps < float("inf") or args.max_frames < 4 or args.max_frames % 2:
        parser.error("Use a positive finite --fps and an even --max-frames of at least 4")
    if not 0 <= args.temperature < float("inf") or args.max_new_tokens < 1:
        parser.error("Use finite nonnegative --temperature and positive --max-new-tokens")
    if any(n is not None and n < 1 for n in (args.workers, args.concurrency,
                                             args.tensor_parallel_size, args.data_parallel_size)):
        parser.error("Worker, concurrency, and parallelism counts must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or args.max_model_len <= args.max_new_tokens:
        parser.error("Use 0 < --gpu-memory-utilization < 1 and --max-model-len > --max-new-tokens")
    if (Path(args.model) / "lora_adapter").is_dir():
        parser.error("Use the merged release model, not an unmerged training export")
    import torch

    try:
        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        configure_parallelism(args, visible_gpus)
    except ValueError as exc:
        parser.error(str(exc))
    with open_output(args.output, run_config(args), args.resume) as (completed, save):
        print(f"Previously completed: {len(completed)}", file=sys.stderr, flush=True)
        rows = read_rows(args.input, completed)
        first = next(rows, None)
        if first is None:
            print("No remaining tasks.", file=sys.stderr)
            return 0
        print(f"Using {args.data_parallel_size * args.tensor_parallel_size}/{visible_gpus} visible GPUs | "
              f"DP={args.data_parallel_size} TP={args.tensor_parallel_size} | "
              f"CPU preprocessing workers={args.workers} | total concurrency={args.concurrency}",
              file=sys.stderr, flush=True)
        counts = asyncio.run(run(args, chain([first], rows), save))
    if counts["error"]:
        print("Some tasks failed. See error records in the output; use --resume to retry.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
