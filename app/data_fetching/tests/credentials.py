import os
from dotenv import load_dotenv

load_dotenv()

print("MODEL_NAME =", os.getenv("MODEL_NAME"))
print("GOOGLE_APPLICATION_CREDENTIALS =", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))


from langchain.chat_models import init_chat_model

llm = init_chat_model(model=os.getenv("MODEL_NAME"), temperature=0.2)
print("LLM initialized successfully!")
