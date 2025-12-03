# Video-CLIP-Indexer
A GUI short-video 'clip' indexer powered by Jina CLIP v2.

## Basic Usage
### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Get a Jina API Key
Get your API key from [Jina AI](https://jina.ai/).

### 3. Run the streamlit GUI
```bash
streamlit run app.py
```

## Parameters
### Text Prompt
You can use a prompt to describe the scene you want to search for. The indexer will return several clips related to it.
### Top N
The number of video clips you want to be returned.
### Cut Sim
Approximately from 0.44 to 0.6. The smaller the number, the longer the video clips (with lower precision).
### Jina API Key
Your Jina API key for accessing the Jina CLIP v2 embedding model.

## How it works
1. The video is processed to extract keyframes based on perceptual hash similarity
2. Each keyframe is converted to base64 and sent to Jina CLIP v2 API
3. The text prompt is also embedded using Jina CLIP v2
4. Cosine similarity is calculated between the text embedding and each frame embedding
5. The top N matching video segments are returned and displayed
