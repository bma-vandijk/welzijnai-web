"""Streamlit web application for health assessment conversations and tracking.

This app provides an interface for conducting health-related conversations,
recording responses, and visualizing health metrics over time using both
text and speech modes.

The app implements:
- User authentication
- Text and speech-based conversation modes
- EQ-5D-5L health questionnaire assessment
- Clinical Frailty Scale (CFS) evaluation
- Historical data visualization
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import os
#import time
import pandas as pd
import streamlit as st
#from audiorecorder import audiorecorder
#from audio_recorder_streamlit import audio_recorder
from groq import Groq
from gtts import *
from streamlit import session_state as ss
from streamlit_autorefresh import st_autorefresh

from src.llm import LLM
from app_utils import *


# Constants
STT_API: bool = True  # Whether to use API for Speech-to-Text
TTS_API: bool = True  # Whether to use API for Text-to-Speech
CONTINUOUS_MIC_RECORDING: bool = True  # Whether to use continuous microphone recording
PILOT_MODE: bool = False  # Whether to use pilot mode
TEXT_STREAM_DELAY: float = 0.05  # Delay between words when streaming text

# Initialize authentication
auth = get_authenticator()
try:
    auth.login(fields={"Form name": "Inloggen", "Username": "Gebruikersnaam", "Password": "Wachtwoord",
                           "Login": "Login"})
except Exception as e:
    st.error(e)



# Main application logic - only runs when user is authenticated
if ss["authentication_status"]:

    @st.cache_resource
    def get_LLM() -> LLM:
        """Initialize and cache LLM instance for conversation handling."""
        
        return LLM()

    @st.cache_resource
    def get_TTS_module():
        """Initialize and cache the speech-to-text pipeline."""
        
        return get_transcriber()

    
    # Initialize core components
    llm = get_LLM()
    recorder = ContinuousMicRecorder()
    
    if not TTS_API:
        transcriber, tts_kwargs = get_TTS_module()

    # Initialize session state variables if not present
    if "question_index" not in ss:
        ss.question_index = 0

    # -- chat history
    if "messages" not in ss:
        ss.messages = []
        ss.messages.append({"role": "assistant", "content": llm.init_conversation[0]})

    # -- llm context
    if "context" not in ss:
        ss.context = ""
        ss.context += f"Begeleider: {llm.init_conversation[0]}"

    # -- text-to-score task
    if "scores" not in ss:
        ss.scores = []

    if "message_count" not in ss:
        ss.message_count = 0
    
    if "disable_start_button" not in ss:
        ss.disable_start_button = False

    if "user_ready" not in ss:
        ss.user_ready = None

    def next_session() -> None:
        """Reset session state and clear cache for a new conversation."""
        
        for key in ss.keys():
            del ss[key]
        st.cache_data.clear()
        llm.conversation_ended = False

    def disable_start_button() -> None:
        """Disable the start button."""
        ss.disable_start_button = True

    def show_history() -> None:
        """Switch to dashboard view by setting conversation end flag."""
        llm.conversation_ended = True

    def display_message(message: str, role: str) -> None:
        """Display a chat message with custom styling based on the role.

        Args:
            message: The text content of the message
            role: Either 'user' or 'assistant' to determine message styling
        """
        style = {
            "user": {
                "bg_color": "#b3c8f7",
                "margin_left": "auto",
                "margin_right": "0",
                "shadow": "1px 1px 5px rgba(0, 0, 0, 0.1)",
            },
            "assistant": {
                "bg_color": "#FFFFFF",
                "margin_left": "0",
                "margin_right": "auto",
                "shadow": "1px 1px 5px rgba(0, 0, 0, 0)",
            }
        }

        role_style = style[role]
        st.markdown(
            f"""
            <div style='
                background-color: {role_style["bg_color"]};
                color: black;
                padding: 8px 12px;
                border-radius: 15px;
                margin: 5px 0;
                display: inline-block;
                max-width: {'70%' if role == 'user' else '100%'};
                word-wrap: break-word;
                font-size: 24px;
                text-align: left;
                margin-left: {role_style["margin_left"]};
                margin-right: {role_style["margin_right"]};
                box-shadow: {role_style["shadow"]};
            '>
                {message}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -- Set html for larger fonts in layout and chat history
    st.html(
        """
        <style>
        .stChatMessage:has(.chat-user) {
            flex-direction: row-reverse;
            font-size: 24px;
            text-align: right;
        }
        [data-testid="stChatMessage"] {
            display: flex;
            margin: 5px 0;
            max-width: 70%;
            width: 70%;
            padding: 10px;
            border-radius: 8px;
            word-wrap: break-word;
            background-color: #FFFFFF;
        }
        </style>
        """
    )

    # Main conversation loop
    if not llm.conversation_ended:
        # Sidebar controls
        with st.sidebar:
            st.write(rf"$\textsf{{\LARGE {ss['name']}}}$")
            auth.logout(r"$\textsf{\large Log uit}$")
            st.markdown("""***""")
            
            # Mode selection
            text = r"$\textsf{\large Tekst}$"
            speech = r"$\textsf{\large Spraak}$"
            
            if PILOT_MODE:
                mode = st.radio(
                    r"$\textsf{\LARGE Besturing}$",
                    options=[speech],
                )
            else:
                mode = st.radio(
                    r"$\textsf{\LARGE Besturing}$",
                    options=[speech, text],
                    disabled=ss.disable_start_button
                )

            st.markdown("""***""")
            
            ss.user_ready = st.button(r"$\textsf{\large Klik hier om het gesprek te beginnen}$",on_click=disable_start_button,type="primary",disabled=ss.disable_start_button)

            if not PILOT_MODE:
                st.markdown("""***""")
                st.write(r"$\textsf{\Large Ga naar}$")
                st.button(r"$\textsf{\large Geschiedenis}$", on_click=show_history)

            

        # Display conversation history
        if ss.disable_start_button:
            st_autorefresh(interval=2, limit=2)
            #st.write(ss.user_ready)
            with st.container(border=False,height=800 if mode == speech else 550):
                    
                for message in ss.messages:
                    #st.write(ss)
                    with st.chat_message(message["role"]):
                        st.html(f"<span class='chat-{message['role']}'></span>")
                        
                        if ss.message_count == 0:
                            display_message(message["content"], message["role"])
                            ss.message_count += 1
                        else:
                            display_message(message["content"], message["role"])
                            
                            if mode == speech and ss.question_index == 0:
                            #st.write(message["content"], message["role"])
                            
                                if not TTS_API:
                                    audio_file = generate_speech_locally(message["content"], "nl", ss["username"])
                                    play_audio_autoplay(audio_file)
                                    
                                else:
                                    audio_file = generate_speech(message["content"], "nl", ss["username"])
                                    play_audio_autoplay(audio_file)
                            

                # Text mode
                if mode == text:
                    st.markdown(
                        """
                        <style>
                        div[data-testid="stMainBlockContainer"]{
                        position:fixed;
                        bottom:20%;
                        padding: 50px;
                        }
                        div[data-testid="stChatInput"] {
                        position:fixed;
                        bottom:10%;
                        padding: 0px;
                        height: 60px;
                        min-height: 40px !important;
                        padding: 0px !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        """
                        <style>
                        div[data-testid="stChatInputTextArea"] {
                        font-size: 50px !important;
                        padding: 0px !important;
                        height: 60px;
                        min-height: 40px !important;
                        line-height: 1.5 !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )  

                    if prompt := st.chat_input("Typ hier uw bericht",disabled=ss.question_index == 7):

                        ss.messages.append(
                            {"role": "user", "content": prompt}
                        )  # -- add to history
                        ss.context += f" Patiënt: {prompt}"  # -- add to llm context

                        with st.chat_message("user"):  # -- display message in chat
                            st.html(f"<span class='chat-user'></span>")
                            display_message(prompt, role="user")

                        # -- check if end of list is reached and do a writeout
                        if ss.question_index == 7:
                            ss.scores.append(
                                classify_eq5d_response(prompt, ss.question_index - 1)
                            )

                            with st.chat_message("Assistant"):
                                # st.write(ss)
                                response = llm_ends_conversation(ss.context)

                                stream_text_gradually(
                                    text=response,
                                    delay=TEXT_STREAM_DELAY,
                                    message_counter=ss.message_count,
                                )

                                st.markdown("""***""")
                                st.button(r"$\textsf{\Large Klik hier om verder te gaan}$")

                            # -- save scores to df
                            save_eq5d_responses(
                                llm.eq_questions, ss.scores, ss.messages, ss["username"]
                            )
                            
                            # -- save conversation
                            save_conversation(ss.messages,ss['username'])

                            llm.conversation_ended = (
                                True  # -- use this for switching to dashboard
                            )

                        # -- if not continue with llm generated EQ-5D questions
                        else:
                            with st.chat_message("Assistant"):
                                response = get_llm_response(
                                    generate_llm_context(ss.context, ss.question_index)
                                )
                                ss.messages.append({"role": "assistant", "content": response})

                                # -- add to history
                                ss.context += (
                                    f" Begeleider: {response}"  # -- add to llm context
                                )

                                stream_text_gradually(
                                    text=response, delay=TEXT_STREAM_DELAY, message_counter=ss.message_count
                                )

                                # -- text-to-score task
                                if (
                                    ss.question_index != 0
                                ):  # -- skip irrelevant first user response
                                    ss.scores.append(
                                        classify_eq5d_response(prompt, ss.question_index - 1)
                                    )
                                ss.question_index += 1

                # STT + text mode
                elif mode == speech:
                
                    # -- hack for awkward positioning record button 
                    style_rec = """<style> 
                    iframe{
                        position: fixed; 
                        bottom: 7%; 
                        left: 55%; 
                        transform: translate(-50%, 50%); 
                        height: 85px; 
                        z-index: 9; 
                        width: 80px !important;
                    }
                    </style>"""
                    
                    # -- hack for awkward positioning record button when conversation is finished
                    style_no_rec = """<style> 
                    iframe{
                        position: fixed; 
                        bottom: 7%; 
                        left: 5%; 
                        transform: translate(-50%, 50%); 
                        height: 85px; 
                        z-index: 9; 
                        width: 80px !important;
                    }
                    </style>""" 

                    if ss.question_index == 7:
                        st.markdown(style_no_rec, unsafe_allow_html=True)
                    else:
                        st.markdown(style_rec, unsafe_allow_html=True)
                    
                    
                    # # Add overlay to hide audiorecorder when conversation is finished, for fully manual record button
                    # if ss.question_index == 5:
                    #     overlay_style = """
                    #     <style>
                    #     .audio-overlay {
                    #         position: fixed;
                    #         bottom: 10%;
                    #         left: 55%;
                    #         transform: translate(-50%, 50%);
                    #         height: 120px;
                    #         width: 350px;
                    #         background-color: white;
                    #         z-index: 10;  /* Higher z-index than the audiorecorder */
                    #     }
                    #     </style>
                    #     <div class="audio-overlay"></div>
                    #     """
                    #     st.markdown(overlay_style, unsafe_allow_html=True)


                    if audio := choose_audiorecorder(CONTINUOUS_MIC_RECORDING):
                        
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

                        if not STT_API:

                            ASR_result = transcriber(
                                os.path.join(
                                    os.getcwd(),
                                    "output",
                                    "audio_stt",
                                    f"{ss['username']}_STT_{id_}.wav",
                                ),
                                generate_kwargs = tts_kwargs
                            )
                            transcript = ASR_result["text"]
                        else:
                            with open(
                                os.path.join(
                                    os.getcwd(),
                                    "output",
                                    "audio_stt",
                                    f"{ss['username']}_STT_{id_}.wav",
                                ),
                                "rb",
                            ) as file:

                                client = Groq()  # -- use Whisper STT via Groq
                                filename = os.getcwd() + f"{ss['username']}_audio{id_}.wav"

                                # -- STT functionality
                                transcription = client.audio.transcriptions.create(
                                    file=(filename, file.read()),
                                    model="whisper-large-v3-turbo",
                                    response_format="verbose_json",
                                )
                                transcript = transcription.text

                        with st.chat_message("user"):  # -- display message in chat
                            st.html(f"<span class='chat-user'></span>")
                            display_message(transcript, role="user")

                            ss.messages.append(
                                {"role": "user", "content": transcript}
                            )  # -- add to history
                            ss.context += f" Patiënt: {transcript}"  # -- add to context

                        # -- from here generate llm questions based on context, but not if end of list is reached
                        if ss.question_index == 7:
                            ss.scores.append(
                                classify_eq5d_response(transcript, ss.question_index - 1)
                            )

                            with st.chat_message("Assistant"):
                                response = llm_ends_conversation(ss.context)
                                
                                stream_text_gradually(
                                    text=response,
                                    delay=TEXT_STREAM_DELAY,
                                    message_counter=ss.message_count,
                                )

                                if not TTS_API:
                                    audio_file = generate_speech_locally(response, "nl", ss["username"])
                                    play_audio_autoplay(audio_file)
                                    
                                else:
                                    # -- TTS locally
                                    audio_file = generate_speech(response, "nl", ss["username"])
                                    play_audio_autoplay(audio_file)

                                st.markdown("""***""")
                                st.button(r"$\textsf{\Large Klik hier om verder te gaan}$")

                            # -- save scores to df
                            save_eq5d_responses(
                                llm.eq_questions, ss.scores, ss.messages, ss["username"]
                            )

                            # -- save conversation
                            save_conversation(ss.messages,ss['username'])

                            llm.conversation_ended = (
                                True  # -- use this for switching to dashboard
                            )

                        # -- continue conversation by voice
                        else:
                            with st.chat_message("Assistant"):
                                response = get_llm_response(
                                    generate_llm_context(ss.context, ss.question_index)
                                )

                                ss.messages.append(
                                    {"role": "assistant", "content": response}
                                )

                                stream_text_gradually(
                                    text=response,
                                    delay=TEXT_STREAM_DELAY,
                                    message_counter=ss.message_count,
                                )

                                if not TTS_API:
                                    # -- TTS API
                                    audio_file = generate_speech_locally(response, "nl", ss["username"])
                                    play_audio_autoplay(audio_file)

                                else:
                                    # -- TTS locally
                                    audio_file = generate_speech(response, "nl", ss["username"])
                                    play_audio_autoplay(audio_file)

                                # -- text-to-score task
                                if ss.question_index != 0:
                                    ss.scores.append(
                                        classify_eq5d_response(
                                            transcript, ss.question_index - 1
                                        )
                                    )

                                ss.question_index += 1

    else:
        
        if PILOT_MODE:
            st.header("Dank voor uw medewerking aan dit experiment. Uw ervaringen met dit systeem helpen ons om dit soort applicaties beter te maken.")

            # Current CFS session assessment only if questionnaire is completed
            if ss.question_index == 7:
                #st.header("Huidige fragiliteit")
                CFS_assessment, score = assess_clinical_frailty(
                    ss.context,
                    os.path.join(os.getcwd(), "src", "CFS.json"),
                )
                #st.write(CFS_assessment)
                #st.write(score)

            # else:
            #     st.header("Huidige fragiliteit")
            #     st.write("Voer eerst een gesprek voor een actuele inschatting van fragiliteit.")

            st.button("Sessie afsluiten en naar volgende gesprek gaan", on_click=next_session) 

        else:
            st.header("Geschiedenis Welzijn")
            
            eq5d_data_path = os.path.join(DATA_OUTPUT_DIR, 'eq5d')
            
            eq5d_files = [f for f in os.listdir(eq5d_data_path) if not f.startswith(".")]
            
            if len(eq5d_files) == 0:
                st.write("Voer meer gesprekken om een geschiedenis van Welzijn te krijgen.")
            
            elif len(eq5d_files) == 1:
                sessions_df_eq5d = pd.read_csv(os.path.join(eq5d_data_path, eq5d_files[0]))
                st.bar_chart(
                sessions_df_eq5d,
                x="eq5d_cat",
                y="scores",
                x_label="EQ-5D categorie",
                y_label="Score",
                color="eq5d_cat",
                use_container_width=False,
                width=800,
                height=400,
                )
            
            else:
                sessions_df_eq5d = concat_session_data("eq5d")
                st.line_chart(
                sessions_df_eq5d,
                x="time_id",
                y="scores",
                x_label="Sessiedatum",
                y_label="EQ-5D score (lager is beter)",
                color="eq5d_cat",
                use_container_width=False,
                width=800,
                height=400,
                )

            st.markdown("""***""")

            st.header("Geschiedenis Fragiliteit")
            
            cfs_data_path = os.path.join(DATA_OUTPUT_DIR, 'cfs')
            
            if len([f for f in os.listdir(cfs_data_path) if f != ".DS_Store"]) == 0:
                st.write("Voer meer gesprekken om een geschiedenis van fragiliteit te krijgen.")
            
            else:
                sessions_df_cfs = concat_session_data("cfs")
                st.line_chart(
                    sessions_df_cfs,
                    x="time_id",
                    y="cfs_cat",
                    x_label="Sessiedatum",
                    y_label="CFS score (lager is beter)",
                    use_container_width=True,
                )

            st.markdown("""***""")
            # Current CFS session assessment only if questionnaire is completed
            if ss.question_index == 7:
                st.header("Huidige fragiliteit")
                CFS_assessment, score = assess_clinical_frailty(
                    ss.context,
                    os.path.join(os.getcwd(), "src", "CFS.json"),
                )
                st.write(CFS_assessment)
                save_cfs_score(score, ss["username"])
                #st.write(score)

                st.markdown("""***""")
                
                # Presents eq5d topic, score, score and chatbot questions
                st.header("Scores gegeven door chatbot")
                score_df = pd.DataFrame(
                    {
                        "topic": llm.eq_questions,
                        "chatbot_questions": [
                            d["content"] for d in ss.messages if d["role"] == "assistant"
                        ][1:],
                        "user_response": [
                            m["content"] for m in ss.messages if m["role"] == "user"
                        ][1:],
                        "chatbot_scores": ss.scores,
                    }
                )
                st.write(score_df)
        
            else:
                st.header("Huidige fragiliteit")
                st.write("Voer eerst een gesprek voor een huidige beoordeling van fragiliteit.")

            st.button("Naar de volgende sessie", on_click=next_session)

elif ss["authentication_status"] is False:
    st.error("Uw gebruikersnaam of wachtwoord klopt niet.")
elif ss["authentication_status"] is None:
    st.warning("Vul alstublieft uw gebruikersnaam en wachtwoord in.")
