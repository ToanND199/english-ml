from utils import *
from dotenv import load_dotenv
load_dotenv()

prompt = "You are teacher of english. You can feedback to student's speaking. Please give some feedback to the following speaking."

context = "hello I'm toan I from Ha Noi"
template_messages=[
    {
        "role": "system",
        "content": f"{prompt}"
    },
    {
        "role": "user",
        "content": f"{context}"
    }
]

response = chat_completions(template_messages)
print(response)