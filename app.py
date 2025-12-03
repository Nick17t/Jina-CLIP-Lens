import streamlit as st
from helper import search_frame, get_keyframes_data, read_video, write_video_clip
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title='Jina CLIP Lens', page_icon='🔍')
st.title('Jina CLIP Lens')

uploaded_file = st.file_uploader('Choose a file')
text_prompt = st.text_input('Text Prompt', '')
topn_value = st.text_input('Top N', '5')
cut_sim_value = st.text_input('Cut Sim', '0.6')
jina_api_key = st.text_input('Jina API Key', type='password')

search_button = st.button('Search')

if search_button:
    if not jina_api_key:
        st.error('Please enter your Jina API Key')
    elif uploaded_file is not None:
        with st.spinner('Processing...'):
            try:
                logger.info("Starting video processing...")
                os.makedirs('tmp_videos', exist_ok=True)
                video_path = 'tmp_videos/' + uploaded_file.name
                with open(video_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                logger.info(f"Video saved to {video_path}")
                
                logger.info("Reading video...")
                video_data, fps = read_video(video_path)
                logger.info(f"Video read: {len(video_data)} frames, {fps} fps")
                
                logger.info("Extracting keyframes...")
                keyframe_data = get_keyframes_data(video_data, float(cut_sim_value))
                logger.info(f"Extracted {len(keyframe_data)} keyframes")
                
                logger.info("Searching frames with Jina API...")
                spans, frames, scores = search_frame(keyframe_data, text_prompt, int(topn_value), jina_api_key)
                logger.info(f"Search complete, found {len(spans)} results")
                
                for i, span in enumerate(spans):
                    save_name = 'tmp_videos/' + str(i) + '_tmp.mp4'
                    clip_frames = video_data[int(span['left']):int(span['right'])]
                    logger.info(f"Writing clip {i}: frames {span['left']} to {span['right']}")
                    write_video_clip(save_name, clip_frames, fps)
                    st.video(save_name)
                    st.caption(f'Score: {scores[i]:.4f}')
                    os.remove(save_name)
                    
                st.success('Done!')
            except Exception as e:
                error_msg = f'Error: {str(e)}'
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(error_msg)
                st.code(traceback.format_exc())
    else:
        st.warning('Please upload a video file')
