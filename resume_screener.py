
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

from docx2pdf import convert
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import docx2txt
from nltk.tokenize import WhitespaceTokenizer

import plotly.graph_objects as go
import plotly.express as px

import chart_studio.plotly as py

warnings.filterwarnings('ignore')
import re
import os
import aspose.words as aw



def find_score(filename,jd):
    doc=aw.Document(filename)
    doc.save(filename.split(".")[0]+".pdf")
    df = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')
    resumeDataSet = df.copy()
    resumeDataSet['cleaned_resume'] = ''
    def cleanResume(resumeText):
        resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
        resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
        return resumeText
        
    resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))


    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        resumeDataSet[i] = le.fit_transform(resumeDataSet[i])


    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    requiredText = resumeDataSet['cleaned_resume'].values
    requiredTarget = resumeDataSet['Category'].values

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)



    X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)


    clf = KNeighborsClassifier(n_neighbors=15)
    clf = clf.fit(X_train, y_train)
    yp = clf.predict(X_test)


    class JobPredictor:
        def __init__(self) -> None:
            self.le = le
            self.word_vectorizer = word_vectorizer
            self.clf = clf

        def predict(self, resume):
            feature = self.word_vectorizer.transform([resume])
            predicted = self.clf.predict(feature)
            resume_position = self.le.inverse_transform(predicted)[0]
            return resume_position

        def predict_proba(self, resume):
            feature = self.word_vectorizer.transform([resume])
            predicted_prob = self.clf.predict_proba(feature)
            return predicted_prob[0]


    job_description = """
    Skills Required:


    • Hands on years of working experience with ETL integration, Core JAVA, Spring Boot and APIs

    • Good knowledge of DB2 or Azure SQL server (experience developing SQL queries)

    • Understanding of File Transfer protocols and processes ie. FTP, SFTP, PGP Encryption

    • Understanding mainframe integration for ETL processing

    • Technical working experience with UNIX shell scripting

    • Knowledge and understanding of Web Services

    • Experience in developing ETL processes (preferably Talend, iWay, DataStage)

    • Experience in writing/creating/updating technical documents

    • Experience in batch job/process scheduling

    • Familiarity with data integration and data streaming, WebSphere MQ and Communication Networks

    • Familiarity with event driven programming concepts

    • Exposure to Data Modelling and Data Architecture


    Roles & Responsibilities:


    • Act as an expert technical resource for problem analysis and solution implementation

    • Work closely with Delivery and Technical Architecture teams, Product Owners and Technical Platform teams to design and develop high quality solutions supporting enterprise architecture and business process improvements that support our business and technical strategies

    • Deal effectively with external Vendors, Business Partners, internal Stakeholders and Management

    • Implement new systems or enhancements including, reviewing programs written by team members, establishing and supporting system test procedures, developing implementation plan, developing the required program and system documentation and ensuring all functionality has been delivered as required

    • Provide post implementation support and training to the Production Support staff on the production processing functionality

    • Support other development areas providing technical expertise, guidance, advice and knowledge transfer to staff and more junior Developers

    • Coordinate and accommodate with a geographically dispersed team

    • Pager rotation mandatory during critical processing times
    """

    with open(jd,"r") as f:
        job_description=f.read()

    resume_position = JobPredictor().predict(job_description)


    text_tokenizer= WhitespaceTokenizer()
    remove_characters= str.maketrans("", "", "±§!@#$%^&*()-_=+[]}{;'\:,./<>?|")
    cv = CountVectorizer()

    resume_docx = docx2txt.process(filename)

    text_docx= [resume_docx, job_description]
    #creating the list of words from the word document
    words_docx_list = text_tokenizer.tokenize(resume_docx)
    #removing speacial charcters from the tokenized words 
    words_docx_list=[s.translate(remove_characters) for s in words_docx_list]
    #giving vectors to the words
    count_docx = cv.fit_transform(text_docx)
    #using the alogorithm, finding the match between the resume/cv and job description
    similarity_score_docx = cosine_similarity(count_docx)
    match_percentage_docx= round((similarity_score_docx[0][1]*100),2)
    # f'Match percentage with the Job description: {match_percentage_docx}'

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = match_percentage_docx,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Match with JD"}))

    fig.write_image("static/percent.jpg")

    job_predictor = JobPredictor()
    resume_position = job_predictor.predict(resume_docx)

    chart_data = pd.DataFrame({
        "position": [cl for cl in job_predictor.le.classes_],
        "match": job_predictor.predict_proba(resume_docx)
    })

    fig = px.bar(chart_data, x="position", y="match",
                    title=f'Resume matched to: {resume_position}')
    fig.write_image("static/match.jpg")

    return [match_percentage_docx,resume_position]

    

