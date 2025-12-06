import os
from groq import Groq
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def ask_gemini(prompt:str)-> str:
    """Send a natural language query to gemini and get back a response."""

    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel("gemini-2.5-flash")
    response = llm.generate_content(prompt)

    return response.text.strip()

def ask_groq(prompt:str) -> str:

    client = Groq(api_key=GROQ_API_KEY)
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()

if __name__=="__main__":
    print(ask_gemini("Explain faiss "))