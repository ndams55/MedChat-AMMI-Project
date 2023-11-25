import os
import json
import pandas as pd
import numpy as np
import nltk
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='fr')
# translator has to be a global variable; See how 



def compute_token(conversation):
    
    x = nltk.tokenize.sent_tokenize(conversation)
    full_sentence = ''
    
    for sentence in x:
        trans_sentence = translator.translate(sentence)
        full_sentence = full_sentence + trans_sentence

    return full_sentence



def translate_csv(dataF, dir):
    #  the dataF has to be uploaded without index columns (no needed)

    col_names = dataF.columns
    
    for column in col_names:
        new_column = '{colum}'+'_en'
        dataF[new_column] = dataF[column].map(lambda x:compute_token(x, translator)) 
        dataF.loc[:, [column]]

    dataF.to_csv(dir, index=False, encoding='utf-8')

    print("------------------The translation process is ended!!!------------------")



def translate_json(dataJ, dir):

    filepath = os.path.join(os.path.dirname(__file__), dir)
    with open(filepath) as i18n_file:
        parsed_json = json.loads(i18n_file.read())
    
    translated_data = []
    for conversation in parsed_json:
        if len(conversation['input']) >= 5000:
            
            input = compute_token(conversation, translator)
            instruction = translator.translate(conversation['instruction'])
            if len(conversation['output']) >= 5000: 
                output = compute_token(conversation, translator)
            else:
                output = translator.translate(conversation['output'])
            datadict = {'instruction': instruction, 'input':input, 'output':output}
        
        else:
            if len(conversation['output']) >= 5000:
                output = compute_token(conversation, translator)
                input = translator.translate(conversation['input'])
                instruction = translator.translate(conversation['instruction'])
                datadict = {'instruction': instruction, 'input':input, 'output':output}
            else:
                data = translator.translate_batch(conversation.values())
            datadict = {'instruction': data[0], 'input':data[1], 'output':data[2] }

        translated_data.append(datadict)
    
    with open(dir, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=3)

    print("------------------The translation process is ended!!!------------------")


