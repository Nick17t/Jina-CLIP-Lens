import streamlit as st
from helper import search_frame, get_keyframes_data, read_video, write_video_clip
import os

st.set_page_config(page_title='Video CLIP Indexer', page_icon='🔍')
st.title('Video CLIP Indexer')

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
            os.makedirs('tmp_videos', exist_ok=True)
            video_path = 'tmp_videos/' + uploaded_file.name
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            video_data, fps = read_video(video_path)
            keyframe_data = get_keyframes_data(video_data, float(cut_sim_value))
            
            try:
                spans, frames, scores = search_frame(keyframe_data, text_prompt, int(topn_value), jina_api_key)
                
                for i, span in enumerate(spans):
                    save_name = 'tmp_videos/' + str(i) + '_tmp.mp4'
                    clip_frames = video_data[int(span['left']):int(span['right'])]
                    write_video_clip(save_name, clip_frames, fps)
                    st.video(save_name)
                    st.caption(f'Score: {scores[i]:.4f}')
                    os.remove(save_name)
                    
                st.success('Done!')
            except Exception as e:
                st.error(f'Error: {str(e)}')
    else:
        st.warning('Please upload a video file')
