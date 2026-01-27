
# Sim-3DAfford

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/mexdyf/Sim-3DAfford)

[![Dataset](https://img.shields.io/badge/Huggingface-Dataset-blue)](https://huggingface.co/datasets/mexdyf/3DAffordData)

## Installation

1. Create the Conda environment:

```
conda env create sim3dafford
```

2. Install python dependency:

```
pip install -r requirements.txt
```

3. Download the dataset `3DAffordData` and unzip to `./dataset`:

```
hf download mexdyf/3DAffordData \
  --repo-type dataset \
  --pattern "Selected_20260127_110943.7z.*" \
  --local-dir .
mkdir dataset
7z x Selected_20260127_110943.7z.001 -o dataset
```

4. Start training:

```
chmod +x train.sh
./train.sh
```
