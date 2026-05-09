import wandb
import glob

wandb.init(project="video-test", name="local-video-upload")

# Log all videos in a folder
video_paths = glob.glob("/Users/tianyaohu/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Desktop - Mac/dev/SFU/RoboSoccer/sw-rl-agent/rsoccer_docker/experiments/SAC/2026-03-08_195840_SSLDribbling-v0_seed0/videos")
for i, path in enumerate(video_paths):
    wandb.log({f"video_{i}": wandb.Video(path, fps=30, format="mp4")}, step=i)

wandb.finish()
