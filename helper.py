import imagehash
from PIL import Image
import numpy as np
import requests
import base64
import io
import cv2
import subprocess
import os


JINA_API_URL = "https://api.jina.ai/v1/embeddings"


def read_video(video_path: str) -> tuple:
    """
    Read video file and return frames as numpy array along with fps.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (frames as numpy array in RGB format, fps)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return np.array(frames), fps


def write_video_clip(output_path: str, frames: np.ndarray, fps: float):
    """
    Write frames to a video file using H.264 codec for web compatibility.
    
    Args:
        output_path: Path for the output video file
        frames: Numpy array of frames in RGB format
        fps: Frames per second
    """
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    
    # Write to temporary file first with cv2
    temp_path = output_path + '.temp.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB back to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    # Convert to H.264 MP4 using ffmpeg for web compatibility
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ], capture_output=True)
    
    # Remove temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)


def get_keyframes_data(video_data: np.ndarray, cut_sim: float):
    """Extract keyframes from video based on perceptual hash similarity."""
    last_hash = imagehash.phash(Image.fromarray(video_data[0]))
    key_frames = [0]
    frame_num = 0
    for each_frame in video_data:
        frame_hash = imagehash.phash(Image.fromarray(each_frame))
        similarity = 1 - (last_hash - frame_hash) / len(frame_hash.hash) ** 2
        if similarity < cut_sim:
            key_frames.append(frame_num)
        frame_num += 1
        last_hash = frame_hash
    video_length = len(video_data)
    key_frames.append(video_length)
    keyframes_data = [((i, key_frames[key_frames.index(i)+1]), video_data[i]) for i in key_frames if i != video_length]
    return keyframes_data


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert a numpy frame to base64 encoded string."""
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_embeddings(inputs: list, api_key: str) -> list:
    """
    Get embeddings from Jina CLIP v2 API.
    
    Args:
        inputs: List of dicts, each with either 'text' or 'image' key
        api_key: Jina API key
    
    Returns:
        List of embedding vectors
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "jina-clip-v2",
        "input": inputs
    }
    
    response = requests.post(JINA_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    # Sort by index to maintain order
    embeddings = sorted(result['data'], key=lambda x: x['index'])
    return [item['embedding'] for item in embeddings]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_frame(keyframe_data: list, prompt: str, topn: int, api_key: str):
    """
    Search for frames matching the text prompt using Jina CLIP v2.
    
    Args:
        keyframe_data: List of ((left, right), frame) tuples
        prompt: Text search query
        topn: Number of top results to return
        api_key: Jina API key
    
    Returns:
        Tuple of (spans, frames, scores)
    """
    # Prepare inputs for Jina API
    # First element is the text prompt, rest are image frames
    inputs = [{"text": prompt}]
    
    # Convert frames to base64 and add to inputs
    for (span, frame) in keyframe_data:
        base64_img = frame_to_base64(frame)
        inputs.append({"image": base64_img})
    
    # Get embeddings from Jina CLIP v2
    embeddings = get_embeddings(inputs, api_key)
    
    # First embedding is the text prompt
    text_embedding = np.array(embeddings[0])
    
    # Rest are image embeddings
    image_embeddings = [np.array(emb) for emb in embeddings[1:]]
    
    # Calculate similarities
    results = []
    for i, ((left, right), frame) in enumerate(keyframe_data):
        score = cosine_similarity(text_embedding, image_embeddings[i])
        results.append({
            'span': {'left': str(left), 'right': str(right)},
            'frame': frame,
            'score': float(score)
        })
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Take top N results
    top_results = results[:topn]
    
    spans = [r['span'] for r in top_results]
    frames = [r['frame'] for r in top_results]
    scores = [r['score'] for r in top_results]
    
    return spans, frames, scores
