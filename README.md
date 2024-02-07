# 🎧 Pod-Helper

![](assets/demo.png)

Pod-Helper is an advanced audio processing tool that goes beyond transcribing at lightning speed. It also offers audio repair capabilities using the MLM (Masked Language Model) objective to ensure your content maintains its quality and vibe.

## Features:
- ⚡ Lightning-fast audio transcription.
- 🛠️ Audio corruption repair.
- ✨ Ensures your content's vibe is just right.

## How to use:

### Prerequisites

- Install TensorRT-LLM for Windows from [tensorrt-llm-windows](https://github.com/NVIDIA/TensorRT-LLM/tree/rel/windows).

### Overview

Pod-Helper utilizes the TensorRT-LLM Whisper example code, primarily from [`examples/whisper`](https://github.com/NVIDIA/TensorRT-LLM/tree/rel/examples/whisper). 

Key components include:
- [`run.py`](./run.py): Performs inference on WAV file(s) using the built TensorRT engines.
- [`app.py`](./app.py): Provides a Gradio interface for microphone input or file upload, utilizing `run.py` modules.

### Usage

Here we show to run main model behind this app [whisper model](https://github.com/openai/whisper/tree/main) in TensorRT-LLM on a single GPU.

### Run

```bash
# If the input file does not have a .wav extension, ffmpeg needs to be installed with the following command:
# apt-get update && apt-get install -y ffmpeg
python3 run.py --name single_wav_test --engine_dir ./tinyrt --input_file assets/1221-135766-0002.wav

# decode a custom audio file and different engine
python3 run.py --name single_wav_test --engine_dir ./tinyrt_no_layernorm --input_file assets/thnx_resampled_16000Hz.wav

# without logger
python3 run.py --log_level none --name single_wav_test --engine_dir ./tinyrt --input_file assets/1221-135766-0002.wav

# Launch the Gradio interface
python3 app.py
```

### Optional: Re-Build TensorRT engine(s)

You can either use the pre-converted models located in the `tinyrt` folder or download the Whisper checkpoint models from [here](https://github.com/openai/whisper/blob/main/whisper/__init__.py#L27-L28).

```bash
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=assets assets/mel_filters.npz https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
wget --directory-prefix=assets https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
# tiny model
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
```

TensorRT-LLM Whisper builds TensorRT engine(s) from the pytorch checkpoint, and saves the engine(s) to the specified directory. Skip this step if you are using the pre-converted models.

```bash
# install requirements first
pip install -r requirements.txt

# Build the tiny model using a single GPU with plugins.
python3 build.py --output_dir tinyrt --use_gpt_attention_plugin --use_gemm_plugin --use_layernorm_plugin  --use_bert_attention_plugin

# Build the tiny model using a single GPU with plugins without layernorm
python3 build.py --output_dir tinyrt_no_layernorm --use_gpt_attention_plugin --use_gemm_plugin  --use_bert_attention_plugin

# Build the tiny model using a single GPU with quantization
python3 build.py --output_dir tinyrt_weight_only --use_gpt_attention_plugin --use_gemm_plugin --use_bert_attention_plugin --use_weight_only
```

<details>
<summary>Gen AI on RTX PCs Developer Contest Entry details (click to toggle)</summary>

Category: General Generative AI Projects category

**Tested on following system:**
- Operating System: Windows 10
  - Version: 22H2 
  - OS Build: 19045.3930
- TensorRT-LLM version: 0.7.1
  - CUDA version: 12.4
  - cuDNN version: 8.9.7.29 
  - GPU: NVIDIA RTX A1000
  - Driver version: 551.23
  - DataType: FP16
  - Python version: 3.10.11

</details>
