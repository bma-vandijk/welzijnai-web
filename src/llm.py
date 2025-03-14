import os
import json
import sys
from typing import Any

from groq import Groq

class LLM:
    def __init__(self) -> None:
        self.client = Groq(
            # This is the default and can be omitted
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.eq_data, self.eq_questions = self._load_data()
        self.question_index = 0
        self.conversation_ended = False
        self.rankings = []
        self.llm_questions = []
        self.init_conversation = ["Goedendag, fijn dat u even tijd vrij hebt voor het doornemen van enkele vragen! Hoe is het met u vandaag?"]
        self.user_answers = []


    def _load_data(self) -> tuple[dict[str, Any], list[str]]:
        with open('src/EQ-5D-5L.json', 'r') as file:
            data = json.load(file)
            questions = list(data["EQ-5D-5L"].keys())
        return data, questions


    def _get_scale_from_category(self) -> str:
        """
        Takes the scale from the json and parses it in a LLM compatible way
        """
        category = self.eq_questions[self.question_index]
        sentences = self.eq_data["EQ-5D-5L"][category]
        scale = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)])
        return scale

