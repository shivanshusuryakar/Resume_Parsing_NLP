from flask import *
from parsing.processing import get_details
from resume_screener import find_score
from info_extractor import getinfo

app = Flask(__name__,template_folder="template",static_folder="static")

def create_table(detail):
    content="""
    <table id='customer'>
    <tr>
    <th>Duration</th>
    <th>Job Role</th>
    <th>Company</th>
    </tr>"""
    for x in detail:
        content+=f"""
        <tr>
        <td>{x[0]}</td>
        <td>{x[1]}</td>
        <td>{x[2]}</td>
        </tr>"""
    content+="</table>"
    return content


@app.route('/')
def home():
	return render_template('index.html',error="")

@app.route('/result',methods=["POST","GET"])
def result():
    if request.method=="POST":
        jd=request.form['jd']
        resume=request.form['resume']
        var=find_score(resume,jd)
        score=var[0]
        profile=var[1]
        if (int)(score)<30:
            return render_template("error.html",score=score)
        return render_template("result.html",score=score,profile=profile,resume=resume)

    else:
        return "<center><h1>Something went wrong</h1></center>"

@app.route("/stats",methods=["POST","GET"])
def stats():
    if request.method=="POST":
        resume=request.form['resume']
        resume=resume.split(".")[0]+".pdf"
        details=get_details(resume)
        content=create_table(details[2])
        contact=getinfo(resume)
        return render_template("statistics.html",content=content,branch=details[0],degree=details[1],phone=contact[0][0],email=contact[1][0])
    else:
        return "<center><h1>Something went wrong</h1></center>"

if __name__ == '__main__':
	app.run(debug=True)

# [['Jun 2018  Present', 'QA Manual Tester', 'Resume Worded, New York, NY'], ['Jan 2015  May 2018', 'QA Manual Tester', 'Growthsi, New York, NY'], ['May 2008  Dec 2014', 'QA Manual Tester (Nov 2011  Dec 2014)', 'RW Capital, San Diego, CA']]