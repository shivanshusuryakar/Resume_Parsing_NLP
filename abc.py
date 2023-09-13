import pandas as pd 

df=pd.read_csv("UpdatedResumeDataSet.csv")
df=df.sample(313)
df = df.iloc[: , 0:]
print(df.head())
# df.to_csv("resume_extracted.csv")