import streamlit as st
import numpy as np
import speech_recognition as sr
from PIL import Image
from groq import Groq
from langdetect import detect
import pandas as pd

# Initialize session state for API key and feedback storage
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

# Set up the Groq API client
def setup_groq_client():
    if not st.session_state.api_key:
        st.error("Please enter your GROQ API Key in the sidebar.")
        return None
    return Groq(api_key=st.session_state.api_key)

# Function to analyze sentiment using Groq API with advanced prompt and multi-lingual support
def analyze_sentiment_with_groq(text):
    client = setup_groq_client()
    if client is None:
        return "Error: Could not set up Groq client.", "gray"
    
    try:
        lang = detect(text)
    except:
        lang = "en"  # Default to English if detection fails
    
    prompt = f"""Analyze the sentiment of the following text in {lang} language. Provide a detailed explanation of the emotional content, 
    considering the context, tone, cultural nuances, and any subtle implications. Then, categorize the overall sentiment into one of these categories:
    'Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative'.

    Additionally, identify the primary emotion (e.g., joy, sadness, anger, fear, surprise, disgust) and its intensity on a scale of 1-10.

    Text ({lang}): "{text}"
    Analysis:"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="mixtral-8x7b-32768",
        temperature=0.3,
        max_tokens=750
    )
    
    analysis = response.choices[0].message.content.strip()
    
    # Extract the sentiment and color-code it
    if "Sentiment: Very Positive" in analysis:
        return analysis, "green"
    elif "Sentiment: Positive" in analysis:
        return analysis, "lightgreen"
    elif "Sentiment: Negative" in analysis:
        return analysis, "orange"
    elif "Sentiment: Very Negative" in analysis:
        return analysis, "red"
    else:
        return analysis, "yellow"

# Function to recognize speech with language detection
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="auto")
            return text
        except sr.UnknownValueError:
            return "Could not understand audio."
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition service."

# Function to analyze image using Groq API
def analyze_image_with_groq(image):
    client = setup_groq_client()
    if client is None:
        return "Error: Could not set up Groq client.", "gray"
    
    prompt = """Analyze the following image description and provide an emotional assessment. 
    Consider the objects, colors, and overall composition described. Identify the likely emotions conveyed and explain your reasoning.
    
    Image description: A photograph showing [INSERT DESCRIPTION HERE]
    
    Analysis:"""
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="mixtral-8x7b-32768",
        temperature=0.3,
        max_tokens=750
    )
    
    analysis = response.choices[0].message.content.strip()
    return analysis, "blue"

# Function to save feedback to session state
def save_feedback(text, analysis, user_feedback):
    st.session_state.feedback_data.append({
        'text': text,
        'analysis': analysis,
        'user_feedback': user_feedback
    })

# Streamlit UI setup
st.title("Advanced Multi-modal Sentiment Analysis Web App")

# Sidebar for API key input
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your GROQ API Key:", type="password")
if api_key:
    st.session_state.api_key = api_key

# Main content
if st.session_state.api_key:
    # Text input section
    text_input = st.text_area("Enter Text (Any Language):")
    if st.button("Analyze Text"):
        result, color = analyze_sentiment_with_groq(text_input)
        st.markdown(f"**Analysis:**")
        st.write(result)
        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>Overall Sentiment</div>", unsafe_allow_html=True)
        
        # Feedback mechanism
        feedback = st.radio("Was this analysis accurate?", ("Yes", "No"))
        if st.button("Submit Feedback"):
            save_feedback(text_input, result, feedback)
            st.success("Thank you for your feedback!")

    # Voice input section
    if st.button("Record Voice"):
        voice_text = recognize_speech()
        st.write(f"Recognized Speech: {voice_text}")
        
        if voice_text and voice_text != "Could not understand audio.":
            result, color = analyze_sentiment_with_groq(voice_text)
            st.markdown(f"**Analysis:**")
            st.write(result)
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>Overall Sentiment</div>", unsafe_allow_html=True)
            
            # Feedback mechanism
            feedback = st.radio("Was this voice analysis accurate?", ("Yes", "No"))
            if st.button("Submit Voice Feedback"):
                save_feedback(voice_text, result, feedback)
                st.success("Thank you for your feedback!")

    # Image input section
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Analyze Image"):
            st.write("Please describe the image:")
            image_description = st.text_area("Image Description")
            if image_description:
                result, color = analyze_image_with_groq(image_description)
                st.markdown(f"**Analysis:**")
                st.write(result)
                st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>Emotional Assessment</div>", unsafe_allow_html=True)

    # Display feedback statistics
    if st.button("Show Feedback Statistics"):
        if st.session_state.feedback_data:
            df = pd.DataFrame(st.session_state.feedback_data)
            
            st.write("Feedback Statistics:")
            st.write(f"Total feedback entries: {len(df)}")
            st.write(f"Accuracy rate: {(df['user_feedback'] == 'Yes').mean():.2%}")
            
            st.bar_chart(df['user_feedback'].value_counts())
        else:
            st.write("No feedback data available yet.")
else:
    st.warning("Please enter your GROQ API Key in the sidebar to use the application.")
