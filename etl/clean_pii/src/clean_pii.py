import json
from pathlib import Path
import subprocess
import click
import ray
import s3_utils
from ffmpeg import FFmpeg
from qwen_vl_utils import process_vision_info
from PIL import Image


@ray.remote
def list_video_files(s3_path: str):
    for video_file in s3_utils.list_files(s3_path):
        yield video_file


@ray.remote
def download_video(s3_path: str):
    import time

    local_dir = Path("tmp").absolute()
    local_dir.mkdir(exist_ok=True)
    local_vid_path = local_dir / Path(s3_path).name
    if not local_vid_path.exists():
        local_path = s3_utils.download_file(s3_path, local_vid_path)
    return str(local_vid_path)


@ray.remote
def extract_frames(video_file, fps=1):
    """
    Extracts frames from the given video file
    """
    video_file = Path(video_file).absolute()
    local_tmp_dir = video_file.parent / video_file.stem
    local_tmp_dir.mkdir(exist_ok=True)
    ffmpeg_cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-vf",
        f"fps={fps}",
        f"{str(local_tmp_dir.absolute())}/frame%05d.png",
    ]
    subprocess.run(
        ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    for frame_file in local_tmp_dir.glob("frame*png"):
        yield str(frame_file)


def load_model_and_processor(model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
    from transformers import (
        Qwen2_5_VLModel,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


@ray.remote(num_gpus=1)
def detect_faces(image_path, model, processor):
    # image_path = "/Users/srimanth/code/ai-work/etl/clean_pii/src/tmp/actioncliptest00002/frame00001.png"
    image = Image.open(image_path)
    image_w, image_h = image.size
    messages = [
        {
            "role": "system",
            "content": "You are a helpful visual agent who gives precise answers in JSON format only",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                    "resized_width": image_w,
                    "resized_height": image_h,
                },
                {
                    "type": "text",
                    "text": "Locate only faces in this image and provide their exact tight bounding box locations",
                },
            ],
        },
    ]
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
    inputs = inputs.to("mps")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Normalize coordinates
    # x_min = 0
    # _, input_height, input_width = (
    #     processor.image_processor(image)["image_grid_thw"][0] * 14
    # )
    # x_min_norm = int(x_min / input_width * width)
    # y_min_norm = int(y_min / input_height * height)
    # x_max_norm = int(x_max / input_width * width)
    # y_max_norm = int(y_max / input_height * height)
    # return x_min_norm, y_min_norm, x_max_norm, y_max_norm
    detected = json.loads(output_text[0][8:-4])
    return {"frame": image_path, "faces": detected}


@click.command()
@click.option(
    "--s3-path", type=str, required=True, help="File containing paths to video files"
)
def clean_pii(s3_path: str):
    assert s3_path is not None and len(s3_path) > 0

    print(f">> Loading model")
    model, processor = load_model_and_processor(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    model_ref = ray.put(model)
    processor_ref = ray.put(processor)
    print(f">> Saved model in Ray")

    s3_path_ref = ray.put(s3_path)
    video_file_refs = list_video_files.remote(s3_path_ref)
    downloaded_video_refs = [download_video.remote(x) for x in video_file_refs]
    videos_frame_refs = [extract_frames.remote(x) for x in downloaded_video_refs]
    for video_frames in videos_frame_refs:
        detected_faces = [
            detect_faces.remote(x, model_ref, processor_ref) for x in video_frames
        ]

    ##
    ## Wait for all job completions
    ##
    waiting_refs = detected_faces
    print(f">>> WAITING for {len(waiting_refs)} entries")
    while len(waiting_refs) > 0:
        ready_refs, waiting_refs = ray.wait(waiting_refs)
        print(f">>> WAITED. Ready={len(ready_refs)}, Waiting={len(waiting_refs)}")
        for ready_ref in ready_refs:
            for e in ready_ref:
                print(f">>> DONE:{ray.get(e)}")


if __name__ == "__main__":
    clean_pii()
