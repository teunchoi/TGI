import argparse
from pathlib import Path

import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_ours import WanVideoPipeline, ModelConfig


ALLOWED_NUM_FRAMES = [25, 33, 65, 81]
DEFAULT_W_EDGE_MAP = {
    25: 2,
    33: 3,
    65: 6,
    81: 8,
}


def build_retro_schedule(num_frames: int, w_edge: int | None = None):
    if num_frames not in ALLOWED_NUM_FRAMES:
        raise ValueError(
            f"num_frames must be one of {ALLOWED_NUM_FRAMES}, got {num_frames}"
        )

    if w_edge is None:
        w_edge = DEFAULT_W_EDGE_MAP[num_frames]

    if not isinstance(w_edge, int):
        raise ValueError(f"w_edge must be an integer, got {type(w_edge).__name__}")

    if w_edge <= 0:
        raise ValueError(f"w_edge must be a positive integer, got {w_edge}")

    if w_edge >= (num_frames / 2):
        raise ValueError(
            f"w_edge must be smaller than half of num_frames. "
            f"Got w_edge={w_edge}, num_frames={num_frames}"
        )

    edge_ratio = w_edge / num_frames
    mid_start = edge_ratio
    mid_end = 1.0 - edge_ratio

    return [
        ("fast", 0.00, round(mid_start, 6)),
        ("slow", round(mid_start, 6), round(mid_end, 6)),
        ("fast", round(mid_end, 6), 1.00),
    ]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run WanVideoPipeline with configurable arguments."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="A freight train moves forward through heavy falling snow.",
        help="Text prompt",
    )
    parser.add_argument(
        "--img_first",
        type=str,
        default="example/first.jpg",
        help="Path to first image",
    )
    parser.add_argument(
        "--img_last",
        type=str,
        default="example/last.jpg",
        help="Path to last image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        choices=ALLOWED_NUM_FRAMES,
        default=81,
        help="Number of frames (allowed: 25, 33, 65, 81)",
    )
    parser.add_argument(
        "--w_edge",
        type=int,
        default=None,
        help=(
            "Width of fast edge region in frames. "
            "Must be a positive integer smaller than half of num_frames. "
            "If omitted, a default value is used to approximate the original schedule."
        ),
    )
    parser.add_argument(
        "--s_edge",
        type=float,
        default=1.06,
        help="s_edge value",
    )
    parser.add_argument(
        "--s_mid",
        type=float,
        default=0.94,
        help="s_mid value",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=0.7,
        help="beta_end value",
    )
    parser.add_argument(
        "--beta_mid",
        type=float,
        default=0.3,
        help="beta_mid value",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/example.mp4",
        help="Output video path",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    retro_schedule = build_retro_schedule(
        num_frames=args.num_frames,
        w_edge=args.w_edge,
    )

    print(f"num_frames={args.num_frames}")
    print(f"w_edge={args.w_edge if args.w_edge is not None else DEFAULT_W_EDGE_MAP[args.num_frames]}")
    print(f"retro_schedule={retro_schedule}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.1-FLF2V-14B-720P",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-FLF2V-14B-720P",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-FLF2V-14B-720P",
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-FLF2V-14B-720P",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                offload_device="cpu",
            ),
        ],
    )
    pipe.enable_vram_management()

    H, W = 480, 864

    video = pipe(
        prompt=args.prompt,
        negative_prompt="",
        input_image=Image.open(args.img_first).convert("RGB"),
        end_image=Image.open(args.img_last).convert("RGB"),
        seed=args.seed,
        tiled=True,
        height=H,
        width=W,
        sigma_shift=16,
        num_frames=args.num_frames,
        retro=True,
        retro_schedule=retro_schedule,
        s_edge=args.s_edge,
        s_mid=args.s_mid,
        retro_smooth=2,
        beta_end=args.beta_end,
        beta_mid=args.beta_mid,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_video(video, str(output_path), fps=15, quality=5)


if __name__ == "__main__":
    main()