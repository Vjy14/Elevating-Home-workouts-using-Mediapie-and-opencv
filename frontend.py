#Frontend

import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
from backend import process_biceupcurl_video, process_pushup_video, process_squat_video
from squats import process_squats_video

def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://cdn.magicdecor.in/com/2023/10/13182750/I-love-gym-M.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed; /* Keep the background image fixed in place */
            height: 100vh; /* Ensure full slide height */
        }
        .title-text {
            color: white;
            text-align: center;
            font-size: 36px;
            padding-top: 50px;  /* Add padding to center title vertically */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='title-text'>SMART AI TRAINER</h1>", unsafe_allow_html=True)

    # Display buttons to start video processing for each exercise
    if st.button("Start Bicep Curl "):
        # Call the backend function to process the bicep curl video
        process_biceupcurl_video()

    if st.button("Start Pushup"):
        # Call the backend function to process the pushup video
        process_pushup_video()

    if st.button("Start Squat"):
        # Call the backend function to process the squat video
        process_squat_video()
    # if st.button("squats"):
    #     process_squats_video()


if __name__ == "__main__":
    main()
