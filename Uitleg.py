import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welkom in de webversie van Welzijn.AI 👋")

st.sidebar.success("Selecteer 'Welzijn.AI.' hierboven om de app te starten.")

st.markdown(
    """
    Welzijn.AI is een app die... **Om de app te starten...** 👈 Inloggen is mogelijk met gebruikersnaam 'admin' en wachtwoord 'admin'.
    
    ### Functies van deze app
    - Paper over eerdere versies en gebruikersevaluaties [arXiv](https://streamlit.io)
    - Bla bla
    
    ### Disclaimer
    - Privacy, juistheid, snelheid, etc.
"""
)

