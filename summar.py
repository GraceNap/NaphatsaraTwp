import streamlit as st
import openai
import json
import pandas as pd
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from newspaper import Article
from nltk.corpus import wordnet
from translate import Translator  # Use translate library for translation

# Download the stopwords resource
nltk.download('stopwords')

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)

def send_prompt(prompt, model="gpt-3.5-turbo"):
    messages_so_far = [
        {"role": "system", "content": "Obtain URL. Text Preprocessing. Summarizing the Article"},
        {'role': 'user', 'content': prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages_so_far
    )
    return response.choices[0].message.content

st.title('Summarization genius')
st.markdown('Input the article URL that you want to summarize.\n'
            'The AI will provide a summary and vocabulary commonly found in TOEIC Exam with meanings and synonyms.')

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

        # Dynamically adjust the length of the input
        max_input_length = 450  # Set the maximum length within the API limit
        total_length = 0
        summary = ""

        for sentence in nltk.sent_tokenize(user_input):
            total_length += len(sentence)
            if total_length <= max_input_length:
                prompt = summary + " " + sentence if total_length > len(sentence) else sentence
                chunk_summary = send_prompt(prompt)
                summary += chunk_summary
            else:
                break

        st.markdown('**AI response:**')
        st.write(summary)

        # Tokenize the article to find important words
        tokens = nltk.word_tokenize(user_input.lower())
        # Remove punctuation and stop words
        tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]

        # Calculate word frequencies
        word_freq = Counter(tokens)

        # Find words by weighted frequency of occurrence in TOEIC exam
        toefl_words = list(set([word for word, freq in word_freq.most_common() if freq > 1]))  # Remove duplicates

        # Translate the summary to Spanish using the translate library
        translator = Translator(to_lang='es')
        summary_spanish = translator.translate(summary)
        st.markdown('**Summary in Spanish:**')
        st.write(summary_spanish)

        # Create a DataFrame to display words, meanings, and synonyms
        vocab_data = {'Words': [], 'Meanings': [], 'Synonyms': []}

        for word in toefl_words:
            # Get meanings and synonyms for each word
            synsets = wordnet.synsets(word)

            if synsets:
                meanings = synsets[0].definition()
                synonyms = [lemma.name() for syn in synsets for lemma in syn.lemmas()]
                vocab_data['Words'].append(word)
                vocab_data['Meanings'].append(meanings)
                vocab_data['Synonyms'].append(', '.join(synonyms))

        vocab_df = pd.DataFrame(vocab_data)
        st.table(vocab_df)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

