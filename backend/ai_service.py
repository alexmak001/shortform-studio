# GPT explanations and quiz generation

import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_topic_explanation(topic):
    response = openai.ChatCompletion.create(
        model="gpt-5-mini-2025-08-07",
        messages=[
            {"role": "system", "content": "You are a friendly AI tutor."},
            {"role": "user", "content": f"Explain {topic} briefly."}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content