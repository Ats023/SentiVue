#Streamlit app
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk 

st.title("SentiVue: A Sentiment Analysis Tool for Product Reviews")
st.markdown("""
Welcome to my streamlit app!  
            
Please ensure your datasets have a column for:  
        1. Product name (type: string)  
        2. Rating (type: int, range: 1-5)  
        3. Textual review body (type: string).  
            
Optionally, you may use the following sample dataset for testing: [Amazon_Electronics_Dataset_Test](https://huggingface.co/datasets/rkf2778/amazon_reviews_mobile_electronics/blob/main/test.csv)
""")

def vaders(data, review, product, rating):
    with st.spinner('Please wait...'):
        df = pd.read_csv(data)
        try:
            df.rename(columns={review:"review",product:"product",rating:"rating"}, inplace=True)
            df = df.head(2000)
            df = df[['product','review','rating']]
        except KeyError:
            st.error("Please enter the correct field names.")
            exit()
        df = df.head(2000)
        df = df[['product','review','rating']]
        
        #VADER SENTIMENT ANALYSIS METHOD
        #VADER: Valence Aware Dictionary and Sentiment Reasoner
        #Bag of words approach- stopwords are removed and each word is combined to a total score

        from nltk.sentiment import SentimentIntensityAnalyzer

        sia = SentimentIntensityAnalyzer()
        res={}
        for i, row in df.iterrows():
            text_review = row['review']
            myid = i
            res[myid] = sia.polarity_scores(text_review)

        vaders = pd.DataFrame(res).T
        vaders = pd.merge(df,vaders,left_index=True,right_index=True)
        # print(vaders.info())
        st.subheader("Modified DataFrame after VADER analysis:")
        st.dataframe(vaders)
        st.subheader("Count of Ratings:")
        st.bar_chart(df['rating'].value_counts().sort_index())

        st.subheader("VADER Scores range per Rating")
        fig1,ax1 = plt.subplots(1,1,figsize=(6,3))
        ax1 = sns.barplot(vaders,x='rating',y='compound')
        ax1.set_title("Compound Score Range per Rating")
        st.pyplot(fig1)

        fig,axs = plt.subplots(1,3,figsize=(15,4))
        sns.barplot(vaders,x='rating',y='pos',color='green',ax=axs[0])
        sns.barplot(vaders,x='rating',y='neu',color='yellow',ax=axs[1])
        sns.barplot(vaders,x='rating',y='neg',color='red',ax=axs[2])
        axs[0].set_title("Positive Score Range per Rating")
        axs[1].set_title("Neutral Score Range per Rating")
        axs[2].set_title("Negative Score Range per Rating")
        st.pyplot(fig)

        #GROUPED DATASET
        # grouped_res = vaders.groupby('product', as_index=False)['compound'].mean(numeric_only=True)
        grouped_res = vaders.groupby('product', as_index=False).agg({'compound':'mean','rating':'mean'})
        vaders_grouped = pd.DataFrame(grouped_res)
        vaders_grouped['entries'] = vaders.groupby('product')['product'].transform('count')

        st.subheader("Grouped Dataset by product names")
        option = st.selectbox(
        'Sort by Compound Score',
        ('No sort', 'From lower to higher', 'From higher to lower'))

        if option=='From lower to higher':
            vaders_grouped_display = vaders_grouped.sort_values(by='compound')
        elif option=='From higher to lower':
            vaders_grouped_display = vaders_grouped.sort_values(by='compound', ascending=False)
        else:
            vaders_grouped_display = vaders_grouped

        st.dataframe(vaders_grouped_display, use_container_width=True)
        # st.scatter_chart(data=vaders,y='compound',size=10)
        index_values = list(vaders_grouped.index)

        chart = alt.Chart(vaders_grouped).mark_circle(size=20).encode(
            x=alt.X('product', axis=alt.Axis(values=index_values)),
            y='compound',
            tooltip=['product', 'compound']
        ).interactive()

        st.altair_chart(chart, theme="streamlit", use_container_width=True)

data = st.file_uploader('Choose a CSV file')

if data is not None:
    product = st.text_input('Field name for Product Name: ', value=None)
    review = st.text_input('Field name for Review Body: ', value=None)
    rating = st.text_input('Field name for Numerical Rating: ', value=None)
    if product!=None and review!=None and rating!=None:
        vaders(data, review, product, rating)
else:
    st.warning('Must upload a file!')