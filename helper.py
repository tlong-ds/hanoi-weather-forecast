from dotenv import load_dotenv
load_dotenv()
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

classification_modeL_cache = {}

llm_model_cache = {}

def get_llm(model_name="gemini-2.5-flash-lite"): 
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.6,
        top_p=0.8,
        top_k=1,
        max_tokens=None,
        max_retries=3,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

prompt = PromptTemplate(
    input_variables=["text", "prediction"],
    template="""
    You are Sen, a sentiment analysis assistant.

    TEXT:
    {text}

    PREDICTION:
    {prediction}

    Given the above text and sentiment prediction, please analyze the user's content.
    """)

eval_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are agent who analyze the sentiment in Twitter's tweets.
    LIST OF TWEETS:
    {text}

    INSTRUCTIONS
    For each tweet in LIST OF TWEETS, analyze the sentiment only without any explanation, in the format: "positive" or "negative"

    EXAMPLE:
    ["positive", "negative", "positive", "negative", "positive"]
    """
)