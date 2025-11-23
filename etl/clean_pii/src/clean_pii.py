from pathlib import Path
import subprocess
import click
import ray
import s3_utils
from ffmpeg import FFmpeg

@ray.remote
def list_video_files(s3_path: str):
    for video_file in  s3_utils.list_files(s3_path):
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
    video_file = Path(video_file).absolute()
    local_tmp_dir = video_file.parent / video_file.stem
    local_tmp_dir.mkdir(exist_ok=True)
    ffmpeg_cmd = ["/opt/homebrew/bin/ffmpeg", "-y", "-i", str(video_file), 
                  "-vf", f"fps={fps}", f"{str(local_tmp_dir.absolute())}/frame%05d.png"]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame_file in local_tmp_dir.glob("frame*png"):
        yield str(frame_file)
    

@click.command()
@click.option("--s3-path", type=str, required=True, help="File containing paths to video files")
def clean_pii(s3_path:str):
    assert  s3_path is not None and len(s3_path) > 0
    s3_path_ref = ray.put(s3_path)
    video_file_refs = list_video_files.remote(s3_path_ref)
    downloaded_video_refs = [download_video.remote(x) for x in video_file_refs]
    frame_refs = [extract_frames.remote(x) for x in downloaded_video_refs]

    ##
    ## Wait for all job completions
    ##
    waiting_refs = frame_refs
    print(f">>> WAITING for {len(waiting_refs)} entries")
    while len(waiting_refs) > 0:
        ready_refs, waiting_refs = ray.wait(waiting_refs)
        print(f">>> WAITED. Ready={len(ready_refs)}, Waiting={len(waiting_refs)}")
        for ready_ref in ready_refs:
            for e in ready_ref:
                print(f">>> DONE:{ray.get(e)}")




if __name__ == "__main__":
    clean_pii()