import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set up stopwords and other constants
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

def fetch_news(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        st.error("Error fetching news.")
        return None

def scrape_article_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_content = ' '.join([para.get_text() for para in paragraphs])
        return article_content
    except requests.RequestException:
        return None

def summarize_article(content):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode("summarize: " + content, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=500, min_length=250, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def display_news(news_data):
    if news_data and 'articles' in news_data:
        for article in news_data['articles']:
            title = article.get('title')
            source = article.get('source', {}).get('name', 'Unknown Source')
            published_at = article.get('publishedAt')
            url = article.get('url')
            url_to_image = article.get('urlToImage')

            content = scrape_article_content(url)

            if content:
                st.subheader(title)
                st.write(f"**Source**: {source}")
                st.write(f"**Published At**: {published_at}")

                # Generate topics from LDA
                topics = lda_topic_modeling(content)
                #st.write(f"**Detected Topics**: {topics}")

                # Generate summary of the article
                summary = summarize_article(content)
                st.write(f"**Summary**: {summary}")

                st.write(f"[Read Full Article]({url})")

                if url_to_image:
                    st.image(url_to_image, caption=title, use_column_width=True)
                else:
                    st.write("No image available.")

                st.write("---")

    else:
        st.write("No articles found.")

def lda_topic_modeling(content):
    tokens = preprocess_text(content)
    dictionary = Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=3)
    topics_str = "; ".join([f"Topic {i + 1}: " + ", ".join([word for word, _ in lda_model.show_topic(i)]) for i in range(len(topics))])
    return topics_str

def main():
    st.title("Summarized News 📰")  # Add title here

    categories = ['Science', 'Technology', 'World', 'Health', 'Entertainment', 'Politics', 'Business', 'Sports', 'Nation', 'Others']
    category = st.selectbox("Select a category", categories)

    language_options = {
        'Arabic': 'ar',
        'German': 'de',
        'English': 'en',
        'Spanish': 'es',
        'French': 'fr',
        'Italian': 'it',
        'Hindi': 'hi',
        'Norwegian': 'no',
        'Portuguese': 'pt',
        'Russian': 'ru',
        'Vietnamese': 'vi',
        'Ukrainian': 'uk',
        'Chinese': 'zh'
    }
    language = st.selectbox("Select a language", list(language_options.keys()))
    language_code = language_options[language]

    if category == 'Others':
        custom_query = st.text_input("Enter your custom query:")
    else:
        custom_query = category

    api_url = f'https://newsapi.org/v2/everything?q={custom_query}&language={language_code}&sortBy=popularity&apiKey=95dcbbbf586f4f4eb07130d3d0b1b197'
    
    news_data = fetch_news(api_url)
    display_news(news_data)

if __name__ == '__main__':
    main()
