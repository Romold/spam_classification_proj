import pandas as pd 
import nltk #NLP package
import matplotlib.pyplot as plt 

df = pd.read_csv('spam.csv' , encoding='latin-1') #read dataset
print(df.shape) #Just because the entire text is dumped into a cell that doesn't mean it is structured 
#our data set is structured

df.drop(columns=['Unnamed: 2' , 'Unnamed: 3' , 'Unnamed: 4' ] , inplace=True) #dropping the unwanted columns 
#inplace true for replace the original df with the one that you finally got as a result when dropping 

df.rename(columns={"v1" : "class" , "v2" : "SMS"} , inplace=True)
print(df.sample(5))

print(df.groupby('class').describe()) #group by to get some data on the grouped by variable 
#describe is to describe what that data is about 
#The top column refers to the duplicates that are available in the dataset and the frequeny refers to the 
#frequency the relevant duplicate has appeared 

df = df.drop_duplicates(keep='first') #drop duplicates and keep the first one of which it was duplicated by 

print(df.groupby('class').describe())

df["length"] = df["SMS"].apply(len) #here we are adding a new column named length into the datafram and 
#the apply method applies the len function to the column sms and counts the word in the SMS including spaces 

 

print(df.head())

df.hist(column='length' , by='class' , bins=50) #get seperate historgrams for each class(spam , ham) by specifying
# by functon inside the function 

#Now the Pre processing is going to start 

from nltk.stem.porter import PorterStemmer #this is for stemming purposes 

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('punkt') #the rules used for stemming 
ps = PorterStemmer() #using the stemmer 

#preprocessing tasks 
#Lower casing  , Tokenization , Removing special characters , Removing stop words and punctuation , Stemmign 

import string

def clean_text(text):
    
    text = text.lower()
    text = nltk.word_tokenize(text) #tokenizing the word based on whitespace 
    
    #isalnum for checking whether they are alphanumerics 
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text: 
        if i not in stopwords.words('english') and i not in string.punctuation: #if not a stop and punctuation
            y.append(i) 
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i)) #stenning function 
        
    return " ".join(y)

df['SMS_cleaned'] = df['SMS'].apply(clean_text) #APPLY cleantext function to every SMS and putting it under the new column

print(df.head())

#Feature extraction


from sklearn.feature_extraction.text import TfidfVectorizer

tf_vec = TfidfVectorizer(max_features=3000) #size of the vocabulary is max_features(based on frequency we will b taking the top 3000)
x = tf_vec.fit_transform(df['SMS_cleaned']).toarray() #transforming from text to numbers and taking the array of it (also x is the input)


y = df['class'].values #this is basically the output of the system 

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=2)

from sklearn.naive_bayes import MultinomialNB #Bayes is the algorithm

model = MultinomialNB()
model.fit(x_train , y_train) #model.fit is saying model please learn 

from sklearn.metrics import accuracy_score #to measure the accuracy 

y_pred = model.predict(x_test) #what is the y for available x data (predicted values )
print(accuracy_score(y_test , y_pred)) #then we compare the 2