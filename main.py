import streamlit as st
import numpy as np
import speech_recognition as sr
from PIL import Image
from groq import Groq
from langdetect import detect
from deepface import DeepFace
import pandas as pd
import cv2

# Initialize session state for feedback storage and API key
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ''

# Set up the Groq API client
def setup_groq_client():
    if not st.session_state.groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        return None
    return Groq(api_key=st.session_state.groq_api_key)

# Function to analyze sentiment using Groq API with advanced prompt and multi-lingual support
def analyze_sentiment_with_groq(text):
    client = setup_groq_client()
    if client is None:
        return "Error: Could not set up Groq client.", "gray"
    
    # Detect language
    try:
        lang = detect(text)
    except:
        lang = "en"  # Default to English if detection fails
    
    prompt = f"""Analyze the sentiment of the following text in {lang} language. Provide a detailed explanation of the emotional content, 
    considering the context, tone, cultural nuances, and any subtle implications. Then, categorize the overall sentiment into one of these categories:
    'Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative'.

    Additionally, identify the primary emotion (e.g., joy, sadness, anger, fear, surprise, disgust) and its intensity on a scale of 1-10.

    Here are some examples in different languages:
    
    Text (English): "I just got promoted at work! I can't believe it!"
    Analysis: This text expresses intense excitement and disbelief about a positive event (a promotion). The use of exclamation marks emphasizes the speaker's enthusiasm.
    Sentiment: Very Positive
    Primary Emotion: Joy (Intensity: 9/10)

    Text (Spanish): "Hoy me siento un poco triste, pero sé que mañana será mejor."
    Analysis: The speaker expresses mild sadness but also shows optimism for the future. This indicates a temporary negative feeling with a positive outlook.
    Sentiment: Neutral
    Primary Emotion: Sadness (Intensity: 4/10)

    Text (French): "Je suis tellement déçu par les résultats des élections."
    Analysis: This statement conveys strong disappointment regarding election results. The use of "tellement" (so much) emphasizes the intensity of the feeling.
    Sentiment: Negative
    Primary Emotion: Disappointment (Intensity: 7/10)

    Now, analyze the following text:
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

# Function to analyze image for emotions using DeepFace
def analyze_image(image):
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # DeepFace expects BGR format, so if the image is RGB, convert it
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Perform the analysis
        result = DeepFace.analyze(img_array, actions=['emotion'])
        emotion = max(result[0]['emotion'], key=result[0]['emotion'].get)
        intensity = result[0]['emotion'][emotion]
        analysis = f"Detected Emotion: {emotion.capitalize()}\nIntensity: {intensity:.2f}"
        return analysis, "blue"
    except Exception as e:
        return f"Error in image analysis: {str(e)}", "gray"

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
api_key_input = st.sidebar.text_input("Enter your Groq API Key:", type="password")
if api_key_input:
    st.session_state.groq_api_key = api_key_input
    st.sidebar.success("API Key set successfully!")

# Main content
if st.session_state.groq_api_key:
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
            result, color = analyze_image(image)
            st.markdown(f"**Analysis:**")
            st.write(result)
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>Detected Emotion</div>", unsafe_allow_html=True)

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
    st.warning("Please enter your Groq API Key in the sidebar to use the app.")
