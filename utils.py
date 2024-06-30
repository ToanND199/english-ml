import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_completions(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        # model="gpt-4o",
        messages = messages
    )
    return response.choices[0].message.content