from huggingface_hub import snapshot_download

local_dir_our = snapshot_download(repo_id="mexdyf/Sim-3DAfford", local_dir="./model")

local_dir = snapshot_download(repo_id="microsoft/TRELLIS-image-large", local_dir="./model/trellis")
