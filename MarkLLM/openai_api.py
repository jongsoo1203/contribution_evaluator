# openai_api.py

import openai

# Set your OpenAI API key (replace 'YOUR_OPENAI_API_KEY' with your actual key)
openai.api_key = ...


def analyze_contribution(essay_text):
    """
    Analyzes the contributions of human and AI in the given essay text.

    Parameters:
    - essay_text (str): The combined essay containing both human and AI content.

    Returns:
    - analysis (str): The analysis of contributions with scores.
    """

    # Define the prompt for the OpenAI API
    prompt = f'''
    The following is an essay that contains both human-written and AI-generated content. Your task is to analyze the essay and quantify the contributions of the human and the AI. Consider word counts, idea units, and weighted significance in your analysis. Provide the final output in the following format:

    Human Contribution:

    Strengths:
    - [List the strengths of the human contribution]
    - Contribution Range: [Provide the percentage ranges based on word count and weighted significance]

    AI Contribution:

    Strengths:
    - [List the strengths of the AI contribution]
    - Contribution Range: [Provide the percentage ranges based on word count and weighted significance]

    Based on the analysis, provide the final scores of contribution for human and AI.

    Essay:
    """{essay_text}"""
    '''

    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
        messages=[
            {"role": "system",
             "content": "You are an assistant that analyzes essays to determine the contributions made by human and AI authors."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )

    # Extract the assistant's reply
    analysis = response['choices'][0]['message']['content']

    return analysis
