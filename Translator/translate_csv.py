import pandas as pd
import numpy as np
import nltk
from deep_translator import GoogleTranslator




def compute_token(conversation, translator):
    
    x = nltk.tokenize.sent_tokenize(conversation)
    full_sentence = ''
    
    for sentence in x:
        fr_sentence = translator.translate(sentence)

        full_sentence = full_sentence + fr_sentence

    return full_sentence


df = pd.read_csv ("/Users/ndjebayidamarisstephanie/Projects/LLM-Project/French-Med-Bot/data/small-data-train.csv", delimiter=',',encoding='utf-8')
#print(df.head(20))
print('___________________________________Columns__________________________________________')
print(df.columns)


# df = df.iloc[20000:]
print(len(df))
# df['disease']=[str(x) for x in df['disease']]
# df['disease_en']=df['disease'].map(lambda x:translator.translate(x)) 


translator = GoogleTranslator(source='auto', target='fr')


df['short_question_en'] = df['short_question'].map(lambda x:compute_token(x, translator)) 
df['short_answer_en'] = df['short_answer'].map(lambda x:compute_token(x, translator)) 
# df['commonMedications_en'] = df['commonMedications'].map(lambda x:translator.translate(x)) 
# df['Symptom_en'] = df['Symptom'].map(lambda x:translator.translate(x)) 


new_dataF = df.loc[:,['short_question_en', 'short_answer_en']]
print('___________________________________New Columns__________________________________________')
print(new_dataF.columns)
print('___________________________________Saving__________________________________________')
new_dataF.to_csv('/Users/ndjebayidamarisstephanie/Projects/LLM-Project/French-Med-Bot/data/ß', index=False, encoding='utf-8')