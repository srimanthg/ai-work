from pathlib import Path
import click
import ray
import s3_utils

@ray.remote
def list_video_files(s3_path: str):
    for video_file in  s3_utils.list_files(s3_path):
        yield video_file


@ray.remote
def download_video(s3_path: str):
    local_dir = Path("tmp")
    local_dir.mkdir(exist_ok=True)
    local_vid_path = local_dir / Path(s3_path).name
    local_path = s3_utils.download_file(s3_path, local_vid_path)
    return local_path

@click.command()
@click.option("--s3-path", type=str, required=True, help="File containing paths to video files")
def clean_pii(s3_path:str):
    assert  s3_path is not None and len(s3_path) > 0
    s3_path_ref = ray.put(s3_path)
    video_file_refs = list_video_files.remote(s3_path_ref)
    downloaded_video_refs = [download_video.remote(x) for x in video_file_refs]
    print(f">>> WAITING for {len(downloaded_video_refs)} videos")
    ready_refs, waiting_refs = ray.wait(downloaded_video_refs)
    for ready_ref in ready_refs:
        print(f">>> DONE:{ray.get(ready_ref)}")
    while len(waiting_refs) > 0:
        print(f">>> WAITED. Ready={len(ready_refs)}, Waiting={len(waiting_refs)}")
        ready_refs, waiting_refs = ray.wait(waiting_refs)
        for ready_ref in ready_refs:
            print(f">>> DONE:{ray.get(ready_ref)}")




if __name__ == "__main__":
    clean_pii()