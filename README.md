# SentiVue: Python NLTK Sentiment Analysis of Product Reviews
### App Link 👉 [SentiVue Streamlit App](https://sentivue.streamlit.app/)
### ▶️ Description:
This project was created for the application of data analytics in real-life scenarios, encompassing data preprocessing/cleaning, NLP, and data visualization. SentiVue processes a dataset comprising products and their corresponding reviews and ratings. It then generates modified tables and charts, providing valuable information about the performance of each product as well as the entire dataset, as is gathered from the textual reviews.
### ▶️ How to run:
1. Download the 'myApp.py' and 'requirements.txt' files and save them to a folder.
2. Create a virtual environment within the folder and activate it:
<pre>
    py -m venv [name of virtual env]
    //wait for the environment to load
    [name of virtual env]\Scripts\activate
</pre>
3. Install required packages:
<pre>
    pip install -r requirements.txt
    //Or install the packages separately using
    pip install [package name]
</pre>
4. Run the streamlit app:
<pre>
    streamlit run myApp.py
</pre>

### ▶️ How it works:
SentiVue requires the user to upload a csv file with at least 3 required fields: 
1. **Product title**
2. **Review**
3. **Rating**
<br>

![image](https://github.com/Ats023/SentiVue/assets/122550503/39487d7c-846a-49c7-8909-e766432ec1aa)
*Initial portion for user input*

It then follows through the steps:

 - **Data Cleaning and Preparation:**
 SentiVue operates only on the abovementioned fields. It renames columns for the purpose of consistency and drops rows with redundant/invalid data.
 
 - **Sentiment Analysis with VADER:**
 It employs the use of python's NLTK and pandas to parse through the table and generate polarity scores (neg, pos, neu, compound) for each of the reviews. It uses VADER (Valence Aware Dictionary Sentiment Reasoner) to determine whether the review is positive, negative, or neutral, and the degree of its intensity. New columns are added to the table and scores corresponding to each review are entered into their respective places.
 
 - **Generation of correlative graphs and tables:**
Interactive charts, and tabular representation of both grouped and ungrouped data are generated and displayed using matplotlib, seaborn, and streamlit charts. SentiVue offers an overall understanding of the products in the dataset, and their individual market percept.

![image](https://github.com/Ats023/SentiVue/assets/122550503/923a2fdc-5b86-4fc6-87ac-a65a77f18d6e)
*Overall dataset score range*

<br>

![image](https://github.com/Ats023/SentiVue/assets/122550503/761d7c1c-d0d9-4794-99de-3caa0d1c9067)
*Grouped dataset*
