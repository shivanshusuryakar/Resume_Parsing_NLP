import re
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
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


def getinfo(filename):
    phoneRegex = re.compile(r'''(
        (\d{3}|\(\d{3}\))? # area code
        (\s|-|\.)? # separator
        (\d{3}) # first 3 digits
        (\s|-|\.) # separator
        (\d{4}) # last 4 digits
        (\s*(ext|x|ext.)\s*(\d{2,5}))? # extension
        )''', re.VERBOSE)



    emailRegex = re.compile(r'''(
        [a-zA-Z0-9._%+-] + #username
        @                   # @symbole
        [a-zA-Z0-9.-] +     # domain
        (\.[a-zA-Z]{2,4})   # dot-something
        )''', re.VERBOSE)
    
    resume_text = ''
  # calling above function and extracting text
    for page in extract_text_from_pdf(filename):
        resume_text += page + ' '

    resume_text = resume_text.encode("ascii", "ignore")
    resume_text = resume_text.decode()
    resume_text=resume_text.replace("Evaluation Only. Created with Aspose.Words. Copyright 2003-2023 Aspose Pty Ltd.","")
    resume_text=resume_text.replace("Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit: https://products.aspose.com/words/","")
    
    text=resume_text
    print()
    phones=[]
    emails=[]
    for groups in phoneRegex.findall(text):
        phoneNum = '-'.join([groups[1], groups[3], groups[5]])
        if groups[8] != '':
            phoneNum += ' x' + groups[8]
        phones.append(phoneNum)

    for groups in emailRegex.findall(text):
        emails.append(groups[0])


    return [phones,emails]


