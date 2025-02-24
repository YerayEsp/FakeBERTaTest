import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def model_loader(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return tokenizer, model

def chunking(news):
    # Ensure the input is within the maximum token limit - ADDENDUM
    max_length = 512
    if len(news.split()) > max_length:
        # Split the news into chunks of 512 tokens or less - ADDENDUM
        news_chunks = []
        for i in range(0, len(news), max_length):
            chunk = news[i:i + max_length]
            news_chunks.append(chunk)
    else:
        news_chunks = [news]
    
    return news_chunks

def predict_news(tokenizer, model, news_chunks):

    # Process each chunk separately -ADDENDUM
    pred_list = []
    for chunk in news_chunks:
        # Tokenize the input text
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length= 512 )

        # Make the prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities
        prediction = torch.argmax(logits, dim=1).item()
        pred_list.append(prediction)

    # Normalization -ADDENDUM
    if len(pred_list) > 1:
        divisor = len(pred_list)
        numerator = 0
        for result in pred_list:
            numerator += result
        prediction = numerator/divisor
        if prediction < 1:
            prediction = 0

    return prediction

def animated_text(text, speed=0.05):
    placeholder = st.empty()  
    for char in text:
        placeholder.text(placeholder.text() + char)  
        time.sleep(speed)  

model_name = "YerayEsp/FakeBERTa"
tokenizer, model = model_loader(model_name)
st.write('**Welcome to FakeBERTa v.1.0.0!**')
st.markdown("<h4 style='color:red;'>Please note: FakeBERTa only accepts news in English.</h4>", unsafe_allow_html=True)



label = 'Please introduce a news article to analyze:\n'
news = st.text_input(label)

if news:
    news_chunks = chunking(news)


    prediction = predict_news(tokenizer, model, news_chunks)

    if prediction == 0:
                animated_text("The article is **fake**.")
    else:
                animated_text("The article is **real**.")
