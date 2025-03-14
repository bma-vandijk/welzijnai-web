"""Utility functions for the EQ-5D questionnaire and Clinical Frailty Scale assessment application.

This module provides utility functions for:
- Authentication
- Speech-to-text and text-to-speech conversion
- LLM interactions
- Data storage and retrieval
- UI components and styling
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import base64
import json
import os
import re
import time

import gtts
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml
import torch

from streamlit import session_state as ss
from yaml.loader import SafeLoader
from transformers import pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from TTS.api import TTS
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
from audiorecorder import audiorecorder

from src.llm import LLM



# Constants
GROQ_MODEL = "llama-3.3-70b-versatile"  # or "llama-3.1-8b-instant" or "llama3-8b-8192" or "llama3-70b-8192"
LOCAL_WHISPER_MODEL = os.path.join(
    os.getcwd(), "whisper-native-elderly-9-dutch"
)  # or "golesheed/whisper-small"
LOCAL_TTS_MODEL = "tts_models/nl/css10/vits"
# or "tts_models/multilingual/multi-dataset/xtts_v2", but larger and slower model.
AUDIO_OUTPUT_DIR = os.path.join(os.getcwd(), "output", "audio_tts")
DATA_OUTPUT_DIR = os.path.join(os.getcwd(), "output", "data")
CONFIG_DIR = os.path.join(os.getcwd(), ".streamlit")
ELDERLY_FRIENDLY_FONT_SIZE = "24px"
TEXT_STREAM_DELAY = 0.02


# Initialize LLM
llm = LLM()


def get_authenticator() -> stauth.Authenticate:
    """
    Initialize and return the authentication module for user login.
    
    Returns:
        stauth.Authenticate: Authentication object configured from the config file
    """
    with open("creds.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Pre-hash all plain text passwords
    stauth.Hasher.hash_passwords(config["credentials"])

    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )


def get_transcriber() -> Tuple[Any, Dict[str, Any]]:
    """Initialize and return a Whisper speech-to-text pipeline.
    
    Sets up a Whisper model for Dutch speech recognition using MPS acceleration
    if available. Configures the model with appropriate parameters for elderly
    speech recognition.
    
    Returns:
        Tuple containing:
            - pipeline: Configured speech recognition pipeline
            - generate_kwargs: Dictionary of generation parameters
    """
    device = "mps"
    torch_dtype = torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        LOCAL_WHISPER_MODEL, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(LOCAL_WHISPER_MODEL)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    generate_kwargs = {
    "return_timestamps": True,
    "language": "dutch",
    }

    return pipe, generate_kwargs

def generate_llm_context(context: str, question_index: int) -> str:
    """Generate context for the LLM conversation by adding the next EQ-5D question.
    
    Appends the next question from the EQ-5D questionnaire to the existing conversation
    context, formatted in Dutch for natural conversation flow.
    
    Args:
        context: Current conversation context including previous interactions
        question_index: Index of the next EQ-5D question to ask
        
    Returns:
        str: Updated context string with the next question prompt appended
    """
    question = f" Beantwoord de laatste reactie beleefd en beknopt en vraag daarna hoe het gaat met mijn {llm.eq_questions[question_index]} in het Nederlands. Hou de vraag simpel."
    return context + question


def get_llm_response(context: str) -> str:
    """Get response from LLM based on conversation context.

    Args:
        context: Current conversation context including system prompt and user message
        
    Returns:
        str: LLM generated response text in Dutch
    """
    chat_completion = llm.client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Je speelt de rol van een ondersteunende begeleider die de EQ-5D-5L-vragenlijst afneemt bij een patiënt in het Nederlands, door middel van een natuurlijk gesprekje. De vragen zijn kort en worden één voor één gesteld in eenvoudige taal. Aan de patient wordt niet gevraagd om een schaal of om de situatie te kwantificeren. Wanneer de patiënt antwoordt, reageer je op een vriendelijke en behulpzame toon. Na je reactie stel je de volgende vraag. Het is niet nodig om toestemming te vragen, maar je spreekt de patient aan met 'u'. Gebruik geen aanduiding van je begeleidersrol zoals 'Begeleider:'",
            },
            {"role": "user", "content": context},
        ],
        model=GROQ_MODEL,
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content.replace("<|eot_id|>", "")


def llm_ends_conversation(context: str) -> str:
    """Generate a conversation closing message from the LLM.
    
    Creates a natural conclusion to the conversation that summarizes the interaction
    without explicitly mentioning the EQ-5D-5L questionnaire.
    
    Args:
        context: Complete conversation context including all interactions
        
    Returns:
        str: Closing message from the LLM in Dutch
    """
    chat_completion = llm.client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Je bent een begeleider die net de EQ-5D-5L-vragenlijst heeft afgenomen. Je krijgt het hele gesprek te zien. Je reageert eerst kort op het laatste antwoord van de patient. Daarna vat je het gesprek heel kort samen in simpele taal. Je noemt de EQ-5D-5L niet expliciet bij naam. Je sluit het gesprek af, maar niet met 'tot ziens'. Je noemt geen dingen over het verbeteren van de situatie van de patient. Je gebruikt geen enters of witregels in je antwoord.",
            },
            {"role": "user", "content": context},
        ],
        model=GROQ_MODEL,
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content.replace("<|eot_id|>", "")

def save_conversation(messages: List[Dict[str, str]], username: str) -> None:
    """Save the conversation history to a CSV file.
    
    Stores the complete conversation including timestamps and user information
    for later analysis and record-keeping.
    
    Args:
        messages: List of message dictionaries containing role and content
        username: Identifier for the user/patient
    """
    df = pd.DataFrame(messages)
    fname = f"{username}_conversation_{datetime.now().isoformat(timespec='seconds')}.csv"
    df.to_csv(os.path.join(os.getcwd(),DATA_OUTPUT_DIR,'conversations',fname))
    



def classify_eq5d_response(user_response: str, question_index: int) -> str:
    """Classify a user's response on the EQ-5D-5L scale.
    
    Uses the LLM to analyze the patient's natural language response and map it
    to the appropriate level (1-5) on the EQ-5D-5L scale for the given dimension.
    
    Args:
        user_response: Patient's natural language response
        question_index: Index of the EQ-5D question being answered
        
    Returns:
        str: Classification score ("1" through "5")
    """
    scale = llm._get_scale_from_category()
    prompt = f"""
    Je taak is om het volgende antwoord te beoordelen op een schaal van 1 tot 5 voor de categorie '{llm.eq_questions[question_index]} uit de EQ-5D-5L'.
    Gebruik de onderstaande schaal:
    {scale}

    Antwoord van de patiënt: "{user_response}"

    Instructies:
    1. Interpreteer de categorie in kwestie breed en analyseer het antwoord van de patiënt zorgvuldig.
    2. Kies het meest passende niveau op de schaal.
    3. Geef alleen het corresponderende cijfer (1, 2, 3, 4, of 5) als antwoord.

    Jouw antwoord:
    """

    chat_completion = llm.client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Je bent een Nederlandse AI-assistent die antwoorden beoordeelt voor de EQ-5D-5L vragenlijst. Je geeft alleen numerieke antwoorden tussen 1 en 5, en geeft nooit extra tekst of uitleg, ook niet als de antwoorden niet duidelijk zijn.",
            },
            {"role": "user", "content": prompt},
        ],
        model=GROQ_MODEL,
        temperature=0,
    )
    return chat_completion.choices[0].message.content.replace("<|eot_id|>", "")


def load_cfs_scale(path: str) -> str:
    """Load and format the Clinical Frailty Scale from a JSON file.
    
    Reads the CFS definitions and formats them into a structured string
    for use in prompts and display.
    
    Args:
        path: Path to the JSON file containing CFS definitions
        
    Returns:
        str: Formatted string containing numbered CFS levels and descriptions
    """
    with open(path, "r") as file:
        data = json.load(file)
        return "\n".join(
            f"{index + 1}. Categorie: {item['category']}\nBeschrijving: {item['description']}\n"
            for index, item in enumerate(data["Clinical_Frailty_Scale"])
        )


def assess_clinical_frailty(conversation: str, cfs_json_path: str) -> Tuple[str, int]:
    """Assess the patient's Clinical Frailty Scale score based on conversation.
    
    Analyzes the complete conversation between patient and assistant to determine
    the appropriate CFS score and provides justification for the assessment.
    
    Args:
        conversation: Complete conversation text between patient and assistant
        cfs_json_path: Path to the JSON file containing CFS definitions
        
    Returns:
        Tuple containing:
            - str: Detailed assessment text with justification
            - int: Numerical CFS score (1-9)
    """
    scale = load_cfs_scale(cfs_json_path)
    prompt = f"""
    Je taak is om de Clinical Frailty van een patiënt te beoordelen aan de hand van een conversatie tussen patiënt en begeleider. Gebruik de onderstaande schaal:

    {scale}

    Conversatie: "{conversation}"

    Instructies:
    1. Analyseer de conversatie zorgvuldig.
    2. Kies het niveau op de schaal dat het best past bij de toestand van de patient.
    3. Geef het corresponderende cijfer (1, 2, 3, 4, 5, 6, 7, 8, of 9) als antwoord plus de naam van de categorie.
    4. Voeg de relevante categorie en een korte motivatie toe.

    Jouw antwoord:
    """

    chat_completion = llm.client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Je bent een AI-assistent die conversaties tussen patiënt en begeleider beoordeelt volgens de Clinical Frailty Scale. Je geeft numerieke antwoorden tussen 1 en 9, en geeft een korte motivatie van het gekozen antwoord.",
            },
            {"role": "user", "content": prompt},
        ],
        model=GROQ_MODEL,
        temperature=0,
    )

    response = chat_completion.choices[0].message.content
    score = int(re.findall(r"\b(\d)\.", response)[0])
    return response.replace("<|eot_id|>", ""), score


def generate_speech(text: str, language: str, username: str) -> str:
    """Convert text to speech using Google Text-to-Speech.
    
    Generates an MP3 file containing the synthesized speech for the given text.
    
    Args:
        text: Text to convert to speech
        language: Language code (e.g., "nl" for Dutch)
        username: Identifier for the user/patient
        
    Returns:
        str: Filename of the generated audio file
    """
    file_name = f"{username}_TTS_{datetime.now().isoformat(timespec='seconds')}.mp3"
    tts = gtts.gTTS(text=text, lang=language, slow=False)
    tts.save(os.path.join(AUDIO_OUTPUT_DIR, file_name))
    return file_name


def generate_speech_locally(text: str, language: str, username: str) -> str:
    """Convert text to speech using local TTS model.
    
    Generates an MP3 file using a locally hosted TTS model for faster processing
    and better quality control.
    
    Args:
        text: Text to convert to speech
        language: Language code (e.g., "nl" for Dutch)
        username: Identifier for the user/patient
        
    Returns:
        str: Filename of the generated audio file
    """
    device = "cpu"
    file_name = f"{username}_TTS_{datetime.now().isoformat(timespec='seconds')}.mp3"

    # Initialize TTS with local model
    tts = TTS(LOCAL_TTS_MODEL).to(device)
    
    tts.tts_to_file(text=text,
                    #speaker_wav=voice_clone,
                    #speaker="Aaron Dreschner",
                    #language=language,
                    file_path=os.path.join(AUDIO_OUTPUT_DIR, file_name),
                    )
    # multilingual model takes language parameter and speaker parameter/voice clone
    
    return file_name


def play_audio_autoplay(filename: str) -> None:
    """Create an HTML audio element with autoplay for Streamlit.
    
    Embeds an audio player in the Streamlit interface that automatically
    starts playing the specified audio file.
    
    Args:
        filename: Name of the audio file to play
    """
    with open(os.path.join(AUDIO_OUTPUT_DIR, filename), "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)


def stream_words(response: str) -> str:
    """Stream words with a delay for elderly-friendly pacing.
    
    Generator function that yields words one at a time with a delay
    between each word for better readability.
    
    Args:
        response: Text to stream word by word
        
    Yields:
        str: Each word with trailing space
    """
    for word in response.split(" "):
        yield word + " "
        time.sleep(TEXT_STREAM_DELAY)


def stream_text_gradually(text: str, message_counter: int, delay: float = 0.1) -> None:
    """Display text gradually with custom styling in Streamlit.
    
    Shows text character by character with elderly-friendly styling
    and customizable delay.
    
    Args:
        text: Text to display
        message_counter: Counter to track message number
        delay: Time delay between characters in seconds (default: 0.1)
    """
    custom_style = f"""
        <style>
        .streamed-text {{
            font-size: {ELDERLY_FRIENDLY_FONT_SIZE};
            color: black;
        }}
        </style>
    """
    st.markdown(custom_style, unsafe_allow_html=True)

    if message_counter == 0:
        st.markdown(f"<div class='streamed-text'>{text}</div>", unsafe_allow_html=True)
    else:
        with st.empty():
            streamed_text = ""
            for char in text:
                streamed_text += char
                st.markdown(
                    f"<div class='streamed-text'>{streamed_text}</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(delay)


def save_cfs_score(score: int, user: str) -> None:
    """Save Clinical Frailty Scale score to CSV file.
    
    Stores the CFS score along with timestamp and user information
    for later analysis and tracking.
    
    Args:
        score: CFS score (1-9)
        user: Identifier for the user/patient
    """
    score_df = pd.DataFrame({"cfs_cat": score}, index=[0])
    score_df.insert(1, "time_id", datetime.now().isoformat(timespec="seconds"))

    filename = f"{user}_score_df_{datetime.now().isoformat(timespec='seconds')}.csv"
    filepath = os.path.join(DATA_OUTPUT_DIR, "cfs", filename)
    score_df.to_csv(filepath)


def concat_session_data(sub_datafolder: str) -> pd.DataFrame:
    """Concatenate all session data from a subfolder.
    
    Combines multiple session data files into a single DataFrame for analysis.
    For EQ-5D data, adjusts scores to zero-based indexing.
    
    Args:
        sub_datafolder: Name of the subfolder containing session data
        
    Returns:
        pd.DataFrame: Combined DataFrame of all session data
    """
    data_path = os.path.join(DATA_OUTPUT_DIR, sub_datafolder)
    sessions = [f for f in os.listdir(data_path) if not f.startswith(".")]

    sessions_df = pd.read_csv(
        os.path.join(data_path, sessions[0]),
        index_col=0,
    )

    for session_file in sessions[1:]:
        session_data = pd.read_csv(
            os.path.join(data_path, session_file),
            index_col=0,
        )
        sessions_df = pd.concat([sessions_df, session_data])

    if sub_datafolder == "eq5d":
        sessions_df["scores"] = sessions_df["scores"].apply(lambda score: score - 1)
        sessions_df["scores"] = pd.to_numeric(
            sessions_df["scores"], downcast="integer", errors="coerce"
        )

    return sessions_df


def save_eq5d_responses(
    questions: List[str],
    scores: List[int],
    messages: List[Dict[str, str]],
    user: str
) -> None:
    """Save EQ-5D responses and scores to CSV file.
    
    Stores the complete EQ-5D assessment including questions, scores,
    and conversation history.
    
    Args:
        questions: List of EQ-5D questions
        scores: List of numerical scores (1-5)
        messages: List of conversation messages
        user: Identifier for the user/patient
    """
    score_df = pd.DataFrame(
        {
            "eq5d_cat": questions,
            "scores": [int(s) for s in scores],
            "questions": [d["content"] for d in messages if d["role"] == "assistant"][
                1:
            ],
        }
    )
    score_df.insert(2, "time_id", datetime.now().isoformat(timespec="seconds"))

    filename = f"{user}_score_df_{datetime.now().isoformat(timespec='seconds')}.csv"
    filepath = os.path.join(DATA_OUTPUT_DIR, "eq5d", filename)
    score_df.to_csv(filepath)

class ContinuousMicRecorder:
    """Record audio from the microphone in a continuous loop.
    
    This class encapsulates the audio recording functionality and maintains its state
    between Streamlit reruns, allowing for proper initialization of the recorder.
    """
    
    def __init__(self, 
                 text="",
                 neutral_color="#6aa36f",
                 recording_color="#fc0303",
                 icon_size="5x", 
                 energy_threshold=0.01, 
                 pause_threshold=4, 
                 sample_rate=41_000, 
                 auto_start=False):
        """Initialize the continuous microphone recorder.
        
        Args:
            text: The text to display on the recorder button
            icon_size: Size of the recorder button icon
            energy_threshold: Minimum audio energy to detect as speech
            pause_threshold: Seconds of silence to consider recording complete
            sample_rate: Audio sample rate in Hz
            auto_start: Whether to start recording automatically
        """
        self.text = text
        self.recording_color = recording_color
        self.neutral_color = neutral_color
        self.icon_size = icon_size
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        self.sample_rate = sample_rate
        self.auto_start = auto_start
    
    def record(self):
        """Record audio from the microphone.
        
        Returns:
            AudioSegment or False: AudioSegment containing the recorded audio if successful,
                                  False if no audio was recorded
        """
        audio_bytes = audio_recorder(
            text=self.text,
            icon_size=self.icon_size,
            energy_threshold=self.energy_threshold,
            pause_threshold=self.pause_threshold,
            sample_rate=self.sample_rate,
            auto_start=self.auto_start
        )
        
        if audio_bytes is None:
            return False
            
        return AudioSegment(data=audio_bytes)

# Add custom button text
    if audio := audiorecorder("🎙️ Opnemen", "⏹️ Stop",
                              start_style={"color":"green","font-size": "30px", "padding": "20px 30px", "border-radius": "10px", "width": "300px", "height": "100px", "border": "4px solid darkgreen"},
                              stop_style={"color":"red","font-size": "30px", "padding": "20px 30px", "border-radius": "10px", "width": "300px", "height": "100px", "border": "4px solid darkred"},
                              show_visualizer=True,):
        id_ = str(datetime.now().isoformat(timespec="seconds"))
        audio.export(
            os.path.join(
                os.getcwd(),
                "output",
                "audio_stt",
                f"{ss['username']}_STT_{id_}.wav",
            ),
            format="wav",
        )  # -- save current file


class ManualAudioRecorder:


    def __init__(self, 
                 start_prompt="🎙️ Opnemen",
                 stop_prompt="⏹️ Stop",
                 start_style={"color":"green","font-size": "30px", "padding": "20px 30px", "border-radius": "10px", "width": "300px", "height": "100px", "border": "4px solid darkgreen"},
                 stop_style={"color":"red","font-size": "30px", "padding": "20px 30px", "border-radius": "10px", "width": "300px", "height": "100px", "border": "4px solid darkred"},
                 show_visualizer=True):
        
        self.start_prompt = start_prompt
        self.stop_prompt = stop_prompt
        self.start_style = start_style
        self.stop_style = stop_style
        self.show_visualizer = show_visualizer
    
    def record(self):
        """Record audio from the microphone.
        
        Returns:
            AudioSegment or False: AudioSegment containing the recorded audio if successful,
                                  False if no audio was recorded
        """
        audio_bytes = audiorecorder(
            start_prompt=self.start_prompt,
            stop_prompt=self.stop_prompt,
            start_style=self.start_style,
            stop_style=self.stop_style,
            show_visualizer=self.show_visualizer
        )
        
        if audio_bytes is None:
            return False
            
        return audio_bytes



def choose_audiorecorder(CONTINUOUS_MIC_RECORDING: bool):
    
    if CONTINUOUS_MIC_RECORDING:

        recorder = ContinuousMicRecorder()
        audio = recorder.record()
        return audio

    else:
        recorder = ManualAudioRecorder()
        audio = recorder.record()
        return audio
                        