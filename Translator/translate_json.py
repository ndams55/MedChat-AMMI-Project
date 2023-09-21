import os
import json
import nltk
from deep_translator import GoogleTranslator

def tokenize_first(conversation, translator):
    x = nltk.tokenize.sent_tokenize(conversation['input'])
    full_sentence = ''
    
    for sentence in x:
        fr_sentence = translator.translate(sentence)

        full_sentence = full_sentence + fr_sentence

    return full_sentence


translator = GoogleTranslator(source='auto', target='fr')

print(translator.translate("Welcome to our tutorial!"))


filepath = os.path.join(os.path.dirname(__file__), "/Users/ndjebayidamarisstephanie/Projects/LLM-Project/Data_Translated/Dataset/HealthCareMagic-100k.json")

with open(filepath) as i18n_file:
    parsed_json = json.loads(i18n_file.read())


print('------------------------------------------EnglishConversation--------------------------------------------')
print(f'The lenght of the data is {len(parsed_json)}')


    

conversation_trans = []
for conversation in parsed_json:
    if len(conversation['input']) >= 5000:
        
        input = tokenize_first(conversation, translator)
        instruction = translator.translate(conversation['instruction'])
        if len(conversation['output']) >= 5000: 
           output = tokenize_first(conversation, translator)
        else:
            output = translator.translate(conversation['output'])
        datadict = {'instruction': instruction, 'input':input, 'output':output}
    
    else:
        if len(conversation['output']) >= 5000:
           output = tokenize_first(conversation, translator)
           input = translator.translate(conversation['input'])
           instruction = translator.translate(conversation['instruction'])
           datadict = {'instruction': instruction, 'input':input, 'output':output}
        else:
            data = translator.translate_batch(conversation.values())
        datadict = {'instruction': data[0], 'input':data[1], 'output':data[2] }

    conversation_trans.append(datadict)
print('------------------------------------------SavingFrenchConversation--------------------------------------------')


with open('/Users/ndjebayidamarisstephanie/Projects/LLM-Project/Data_Translated/Dataset/trans_HealthCareMagic-100k.json', 'w', encoding='utf-8') as f:
    json.dump(conversation_trans, f, ensure_ascii=False, indent=3)


# import nltk
# nltk.download('punkt')


# x = '''hI, I have no question for you. Just an attitude as to how pain management is being, or should I say not being done. Because of a handful of people who CHOOSE TO Abuse drugs, who really gets punished? Spinal Stenosis and 2 back surgeries, the last the titanium rods, plates, and screws were inserted from L2 thru L5. 
#     Stopped the pain for about 2 years, or at least at a tolerable level. Then Lortabs helped. Sent to pain clinic for epidurals. Epidurals on me were extremely painful. Scale of 1-10 they were a 20. (Not Kidding) Quality of life was going fast. couldn t do or go because the pain had grown to the point that I couldn t eat or sleep. 
#     Pain was 24 hours a day. Told by the pain Dr. to stay on the Lortab and come to the pain clinic every 4 or 5 weeks for epidurals. Epidurals were not only painful but I got very little relief if any from them. Finally PCP started me on oxycontin and Fentynal patch. I was much better for awile. I was taught that in so called pain management you would gradually increase dose until you knew the patient was experiencing some quality of life. 
#     I reached rock bottom after telling PCP my quality of life was slipping away. Finally I asked for an increase in pain med as pain was now 24 hours a day. I could not function. I was told no, and referred to pain clinic for more epidurals and Narcotics. The day I went in to office for routine followup and new perscriptions I was ask for a urine specimen. After all I d been thru over the last 8 to 10 years, all the operations and all the pain. 
#     The loss of any quality of life. I am in my 70 s and lost all those Golden years, 10 of them. Asking for a urine specimen. I felt made ME the criminal. I decided right then I asked to be titrated off the narcotic meds. I will die with all the pain before I will let anyone treat me like a criminal. So you go on breaking your solom oath by hurting patients . Doctors swore to do no harm. A large % of those being hurt and are suffuring are the geriatric population. 
#     I feel Obama has been behind a lot of this but you know I believe a lot of the fault falls on every Doctor, Nurse, and everyone working in health care. STAND UP FOR YOUR PATIENTS. yOU ALL ARE JUST LIKE THE POLITICIANS. tO SAVE THEIR OWN HIDES THEY JUST CLOSE THEIR EYES TO WHAT IS HAPPINING TO MEDICINE IN THIS COUNTRY AND SAY NOTHING. tHE PUBLIC CAN T EVEN GET TOGETHER AND STAND UP FOR WHAT IS RIGHT. The old time Doctor s predicted It would be like this. They said if the HMO went through it would be the death of our wonderful medical care in this country. Many have passed but they were right on. 
#     They realized that there would not be anymore decent medical care. The patient is really obsolete. No more real check ups. no more time to explain your problem. No more real examinations Oh, there is on paper or in the computer so health care workers, even Doctors just make sure they cover their butts, to the point of out right lying!!! Make it look like things were done just to collect the fees. The lawyers are partially to blame also with their ambulance chasing and advertising for frivolous malpractice suits. Rep. and Dems. just sit on their hands. Political correctness has run amuck.
#     I thoroughly loved working in the medical world at one time'''

# x = nltk.tokenize.sent_tokenize(x)
# full_sentence = ''
# for sentence in x:

#     print('------------------------------------------EnglishConversation--------------------------------------------')
#     print(sentence) 
#     fr_sentence = translator.translate(sentence)
#     print('------------------------------------------SavingFrenchConversation--------------------------------------------')
#     print(fr_sentence)
#     full_sentence = full_sentence + fr_sentence

# print(full_sentence)