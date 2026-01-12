import os
from openai import OpenAI

# Initialize OpenAI Client
# Ensure OPENAI_API_KEY is set in your environment
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def query_llm(prompt, api_key,model="gpt-5-mini",):
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert diagnostician minimizing cost."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return "STOP" # Fallback