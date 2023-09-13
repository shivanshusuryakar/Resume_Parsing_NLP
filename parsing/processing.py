
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import spacy
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
import numpy as np
import nltk
import pandas as pd
import dateutil
import re
import json
from pretty_html_table import build_table
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from spacy.matcher import PhraseMatcher
from spacy.lang.en import English

nlp=English()
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))


# PDF to text Conversion
def get_details(filename):
  global nlp
  def extract_text_from_pdf(pdf_path):
      with open(pdf_path, 'rb') as fh:
          # iterate over all pages of PDF document
          for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
              # creating a resoure manager
              resource_manager = PDFResourceManager()
              
              # create a file handle
              fake_file_handle = io.StringIO()
              
              # creating a text converter object
              converter = TextConverter(
                                  resource_manager, 
                                  fake_file_handle, 
                                  codec='utf-8', 
                                  laparams=LAParams()
                          )

              # creating a page interpreter
              page_interpreter = PDFPageInterpreter(
                                  resource_manager, 
                                  converter
                              )

              # process current page
              page_interpreter.process_page(page)
              
              # extract text
              text = fake_file_handle.getvalue()
              yield text

              # close open handles
              converter.close()
              fake_file_handle.close()


  resume_text = ''
  # calling above function and extracting text
  for page in extract_text_from_pdf(filename):
      resume_text += page + ' '


  #removing unwanted unicode characters
  resume_text = resume_text.encode("ascii", "ignore")
  resume_text = resume_text.decode()
  resume_text=resume_text.replace("Evaluation Only. Created with Aspose.Words. Copyright 2003-2023 Aspose Pty Ltd.","")
  resume_text=resume_text.replace("Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/","")
  # Pre- Processing

  #function for tokenizing
  def tokenize(text):
    #using spacy
    doc = nlp(text)
    word_tokens = []
    for token in doc:
      word_tokens.append(str(token))
    return word_tokens

  #function for removing stop words and special characters. also converting tokens to lowercase
  def stop_words(word_tokens):
    #using nltk
    special_char='\"[,@_!$%^&*()<>?/\|}{~:]#.-\''
    removed_stop_words = [word.lower() for word in word_tokens if word not in stops and str(word) not in special_char and not (ord(str(word)[0]) >= 0 and ord(str(word)[0]) <= 32)] 
    return removed_stop_words

  def preProcessess(text):
    #removing \n
    text = text.replace('\n', ' ')
    #word tokenzing
    word_tokens=tokenize(text)
    #removal of stopwords and special characters
    removed_stop_words = stop_words(word_tokens)
    return removed_stop_words

  preProcessedTokens = preProcessess(resume_text)


  # Model 
  # Custom NER model

  df = pd.read_json('job-titles.json')
  df=df.rename(columns={'job-titles': 'text'})
  df['tag']="job-role"

  df_companies=pd.read_csv("companies.csv")

  rows=[]
  for i in range(len(df_companies)):
    row = {"text": df_companies.loc[i, "name"], "tag":"organization"}
    rows.append(row)
  df = df.append(rows, ignore_index=True)




  df = df.sample(frac=1).reset_index(drop=True)  # shuffle the dataset
  X = df.text
  y = df.tag
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

  tag_model = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', MultinomialNB()),
                ])
  tag_model.fit(X_train, y_train)



  y_pred = tag_model.predict(X_test)
  accuracy=accuracy_score(y_pred, y_test)


  def isDate(string, fuzzy=False):
      try: 
          dateutil.parser.parse(string, fuzzy=fuzzy)
          return True

      except ValueError:
          return False
      

  nlp=English()

  # A complete sentence or a link cannot be a header
  def isHeader(text):
    # A complete sentence contains at least one subject, one predicate, one object, and closes with punctuation. 
    # Subject and object are almost always nouns, and the predicate is always a verb.
    text=nlp(text)
    has_noun = 2
    has_verb = 1
    for token in text:
      if token.pos_ in ["NOUN", "PROPN", "PRON"]:
          has_noun -= 1
      elif token.pos_ == "VERB":
          has_verb -= 1
    if has_noun < 1 and has_verb < 1:
      return False
    # check for link
    pattern=r'(?P<url>https?://[^\s]+)'
    links = re.findall(pattern, str(text))
    if links:
      return False
    return True

  # splits the string at each match of: '|', '@' or at
  def partitionExperienceSubheader(string):
    pattern = r'\||@|\bat\b,'
    result = filter(None, re.split(pattern, string))
    return list(result)

  test_strings = [
      "Resume Worded | June 2022 - Present",
      "App developer at Resume Worded",
      "App developer @ Resume Worded",
      "App developer @Resume Worded",
      "Link: https://www.geeksforgeeks.org/",
      "Nurturing Lives | Feb 2020",
      "K. J. Somaiya College of Engineering",
  ]



  headers_experience = (
          'career profile',
          'employment history',
          'work history',
          'work experience',
          'experience',
          'experiences',
          'professional experience',
          'professional background',
          'additional experience',
          'career related experience',
          'related experience',
          'programming experience',
          'freelance',
          'freelance experience',
          'army experience',
          'military experience',
          'military background',
  )
  headers_volunteering = (
          "volunteer",
          "volunteering",
  )
  headers_education = (
          'academic background',
          'academic experience',
          'experiences',
          'programs',
          'courses',
          'related courses',
          'education',
          'qualifications',
          'educational background',
          'educational qualifications',
          'educational training',
          'education and training',
          'training',
          'academic training',
          'professional training',
          'course project experience',
          'related course projects',
          'internship experience',
          'internships',
          'apprenticeships',
          'college activities',
          'certifications',
          'special training',
      )
  headers_skills = (
          'credentials',
          'areas of experience',
          'areas of expertise',
          'areas of knowledge',
          'skills',
          "other skills",
          "other abilities",
          'career related skills',
          'professional skills',
          'specialized skills',
          'technical skills',
          'computer skills',
          'personal skills',
          'interpersonal skills'
          'knowledge',        
          'technologies',
          'technical experience',
          'proficiencies',
          'languages',
          'language competencies and skills',
          'programming languages',
          'competencies'
      )
  headers_projects = (
      'projects',
      'personal projects',
      'academic projects',
      'freelance projects',
  )
  headers_achievements = (
      "other achievements",
      "other accomplishments",
      "previous achievements",
      "previous accomplishments",
      "more achievements",
      "more accomplishments"
  )
  headers_contact = (
      "contact",
      "Reach me",
      "connect",
  )

  # cosine similarity 

  def getSimilarity(all_values, user_values, threshold, flag):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_values)
    filtered_values = []
    score = 0

    for skill in user_values:
      test_word_vector = vectorizer.transform([skill])
      similarities = cosine_similarity(test_word_vector, vectorizer.transform(all_values))
      if similarities.max() > threshold:
              if flag==1:
                filtered_values.append(skill)
              else:
                score = score + similarities.max()

    if flag==1:
      return filtered_values
    else:
      return score


  # phrase match

  def getMatcher(file,col,file_flag):
    if file_flag==1:
      df = pd.read_csv(file)
    else:
      df = pd.read_excel(file)

    column = df[col]
    df = pd.DataFrame(column)
    all_values = df[col].astype(str).values.tolist()
    all_values_lower = [x.lower() for x in all_values]
    nlp=English()
    phrases = all_values_lower
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(phrase) for phrase in phrases]
    matcher.add("PHRASES", None, *patterns)
    return matcher

  def getMatchedPhrases(doc,matcher):
    matches = matcher(doc)
    values=[]
    for match_id, start, end in matches:         
        matched_phrase = doc[start:end]
        values.append(str(matched_phrase))
    return values



  class ResumeParser:

    def __init__(self, text):
      self.resume_text = text
      self.lines = text.split("\n")
    
    resume_sections = {
        'name': {},
        'experience': {},
        'volunteering': {},
        'education': {},
        'skills': {},
        'projects': {},
        'achievements': {},
        'contact': {}
    }
    header_indices = {}

    #Obtain line nos of Headers 
    def getHeaderIndices(self):
      for i, line in enumerate(self.lines):
        if len(line):
          #Header always starts in Uppercase
          if line[0].isupper():
            #The line is a header if it matches the required section headers and isn't a complete sentence
            if [item for item in headers_experience if line.lower().startswith(item)] and isHeader(line):
              self.header_indices[i]="experience"
            elif [item for item in headers_volunteering if line.lower().startswith(item)] and isHeader(line):
              self.header_indices[i]="volunteering"
            elif [item for item in headers_education if line.lower().startswith(item)] and isHeader(line):
              self.header_indices[i]="education"
            elif [item for item in headers_skills if line.lower().startswith(item)] and isHeader(line):
              self.header_indices[i]="skills"
            elif [item for item in headers_projects if line.lower().startswith(item)] and isHeader(line):
              self.header_indices[i]="projects"
            elif [item for item in headers_achievements  if line.lower().startswith(item)] and isHeader(line):
              self.header_indices[i]="achievements"
            elif [item for item in headers_contact if line.lower().startswith(item)] and isHeader(line):
              self.header_indices[i]="contact"

    #Obtain raw text for different resume sections using header indices/linenos
    def getRawResumeSections(self):
      no_lines = len(self.lines)
      header_linenos = list(self.header_indices.keys())
      list_last_index = len(header_linenos)-1
      for counter, lineno in enumerate(header_linenos):
        start_index = lineno+1
        if (counter<list_last_index):
          end_index = header_linenos[counter+1]
        else:
          end_index = no_lines
        self.resume_sections[self.header_indices[lineno]]=self.lines[start_index:end_index]

    def getExperience(self, section_string):
      if not bool(self.resume_sections[section_string]):
        return
      text = self.resume_sections[section_string]
      self.resume_sections[section_string] = [] #list of dictionaries for details of experiences
      exp={"description":""}
      go_to_next=False
      for i, line in enumerate(text):
        if len(line):
          if line[0].isupper():
            # if line isn't a complete sentence, it can be either org, job role or date
            if isHeader(line):
              line_parts = partitionExperienceSubheader(line)
              for line_part in line_parts:
                if isDate(str(nlp(line_part)[0:2])):
                  exp['duration']=line_part
                else:
                  if go_to_next:
                    self.resume_sections[section_string].append(exp) #add to list before moving to next
                    exp={"description":""}
                    go_to_next=False
                  # if not a date, apply model to predict tag
                  exp[tag_model.predict([line_part])[0]]=line_part
            else:
              exp["description"]+=line
              go_to_next=True #Exp description is written at last, after which next experience appears
          else:
              exp["description"]+=line
              go_to_next=True
      self.resume_sections[section_string].append(exp)

    
    def getEducation(self):
        parsed_resume_edu=[]
    
        matcher_b = getMatcher('Major_or_Branch.csv','Major',1)
        matcher_d = getMatcher('Academic Degrees.xlsx','Degree',2)

        # Test the matcher
        for edu in self.resume_sections["education"]:
          
          parsed_branch_degree={}
          edu = edu.lower()
          edu_new = edu.replace(".", "")

          doc = nlp(edu_new)
          
          degrees = getMatchedPhrases(doc,matcher_d)
          if len(degrees)!=0:
            parsed_branch_degree["degree"] = degrees

          branches = getMatchedPhrases(doc,matcher_b)
          if len(branches)!=0:
            parsed_branch_degree["branch"] = branches
         

          if len(parsed_branch_degree)!=0:
            parsed_resume_edu.append(parsed_branch_degree)    
        return parsed_resume_edu      

    #get skills from resume
    def getSkills(self):
          file ='Technology Skills.csv'
          df = pd.read_csv(file)
          column = df['Example']
          df = pd.DataFrame(column)
          all_skills = df['Example'].astype(str).values.tolist()
          all_skills_lower = [x.lower() for x in all_skills]

          resume_skills =""

          for skill in self.resume_sections["skills"]:
            resume_skills =  resume_skills + " "+skill

          resume_skills =  resume_skills.encode("ascii", "ignore")
          resume_skills =  resume_skills.decode()
          resume_skills_tokens = preProcessess(resume_skills)
        
          return getSimilarity(all_skills_lower, resume_skills_tokens, 0.6, 1) 

    #method to parse the entire resume - calls all the above methods of the class
    def getResumeData(self):
      self.getHeaderIndices()
      if len(self.header_indices)!=0:
        self.getRawResumeSections()
        self.getExperience("experience")
        self.getExperience("volunteering")
        self.parsed_resume_education=self.getEducation()
        self.parsed_resume_skills = self.getSkills()
      return self.resume_sections
    

  obj = ResumeParser(resume_text)
  obj.getResumeData()

  def experience():
    return json.dumps(obj.resume_sections["experience"], indent=4, sort_keys=True)
  # print(json.dumps(obj.resume_sections["volunteering"], indent=4, sort_keys=True))

  def education():
    return json.dumps(obj.parsed_resume_education, indent=4, sort_keys=True)
  # print(json.dumps(obj.parsed_resume_skills, indent=4, sort_keys=True))

  x=experience()
  x=x.replace("\n","")
  x=x.strip()

  y=education()
  y=y.replace("\n","")
  y=y.strip()
  xxx= [x,y]

  a=json.loads(xxx[1])
  try:
    branch=(a[0]["branch"][0])
  except:
    branch=''

  try:
    degree=(a[0]["degree"][0])
  except:
    degree=''

  b=json.loads(xxx[0])
  exp_data=[]
  for i in range(len(b)):
    x=[]
    try:
      x.append(b[i]['duration'])
      x.append(b[i]['job-role'])
      x.append(b[i]['organization'])
      exp_data.append(x)
    except:
      continue

  return [branch,degree,exp_data]

