import streamlit as st
import openai
import json
import pandas as pd
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from newspaper import Article
from translate import Translator

# Download the stopwords resource
nltk.download('stopwords')
nltk.download('punkt') 

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

# client = openai.OpenAI(api_key=user_api_key)
openai.api_key = user_api_key

def send_prompt(prompt, model="gpt-3.5-turbo"):
    messages_so_far = [
        {"role": "system", "content": "Obtain URL. Text Preprocessing. Summarizing the Article"},
        {'role': 'user', 'content': prompt},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages_so_far
    )
    return response.choices[0].message.content

def process_input(input_text, model="gpt-3.5-turbo", max_chunk_length=1000):
    chunks = [input_text[i:i + max_chunk_length] for i in range(0, len(input_text), max_chunk_length)]
    processed_chunks = []

    for chunk in chunks:
        messages_so_far = [
            {"role": "system", "content": "Obtain URL. Text Preprocessing. Summarizing the Article"},
            {'role': 'user', 'content': chunk},
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages_so_far
        )
        processed_chunks.append(response.choices[0].message.content)

    return " ".join(processed_chunks)

st.title('Summarization genius')
st.markdown('Input the English article URL that you want to summarize.\n'
            'The AI will provide a summary in English and in Spanish, also vocabulary commonly found in TOEIC C1 level with meanings and synonyms.')

url_input = st.text_input("Enter the article URL:", "Your URL here")

# submit button after URL input
if st.button('Submit'):
    try:
        # Obtain data from the URL
        url = Article(url_input)
        url.download()
        url.parse()

        # Text Preprocessing
        user_input = url.text

        # Summarize the article using GPT-3.5-turbo
        messages_so_far = [
            {"role": "system", "content": "Obtain URL. Text Preprocessing. Summarizing the Article"},
            {'role': 'user', 'content': user_input},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages_so_far
        )

        st.markdown('**English summary:**')
        summary = response.choices[0].message.content
        st.write(summary)

        translator = Translator(to_lang='es')
        summary_spanish = translator.translate(summary)
        st.markdown('**Summary in Spanish:**')
        st.write(summary_spanish)



        


    

        # Tokenize the article to find important words
        tokens = nltk.word_tokenize(user_input.lower())
        # Remove punctuation and stop words
        tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]

        # Calculate word frequencies
        word_freq = Counter(tokens)

        # Find words by weighted frequency of occurrence in TOEIC exam
        toefl_words = list(set([word for word, freq in word_freq.most_common() if freq > 1]))  # Remove duplicates

        # Translate the summary to Spanish using the translate library
        # translator = Translator(to_lang='es')
        # summary_spanish = translator.translate(summary)
        # st.markdown('**Summary in Spanish:**')
        # st.write(summary_spanish)

        # Create a DataFrame to display words, meanings, and synonyms
        vocab_data = {'Words': [], 'Meanings': [], 'Synonyms': []}

        for word in toefl_words:
            # Get meanings and synonyms for each word
            synsets = nltk.corpus.wordnet.synsets(word)

            if synsets:
                meanings = synsets[0].definition()
                synonyms = [lemma.name() for syn in synsets for lemma in syn.lemmas()]
                vocab_data['Words'].append(word)
                vocab_data['Meanings'].append(meanings)
                if synonyms in vocab_data['Synonyms']:
                    pass
                elif synonyms not in vocab_data['Synonyms']:
                     vocab_data['Synonyms'].append(', '.join(synonyms))

        vocab_df = pd.DataFrame(vocab_data)
        st.table(vocab_df)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
