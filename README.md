# AI Image Generator

### 1. download models from [civitai.com](https://civitai.com)

```
wget https://civitai.com/api/download/models/130803 --content-disposition
```

### 2. install virtual environment

```
py -m venv env
```

### 3. install packages

```
pip install -q transformers==4.31.0
pip install -q accelerate==0.21.0
pip install -q diffusers==0.20.0
pip install -q huggingface_hub==0.16.4
pip install -q omegaconf==2.3.0
```

### 4. convert stable diffusion models to diffuser

```
python convert_original_stable_diffusion_to_diffusers.py
    --checkpoint_path ChineseLandscapeArt_v10.safetensors
    --dump_path ChineseLandscapeArt_v10/
    --from_safetensors
```

### 5. run main.py

```
py main.py
```
