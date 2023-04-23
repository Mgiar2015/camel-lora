
import json
import requests 
import os 
import random  
PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"  
instruction_one_persona_a = "Generate the next utterence in the interesting conversation between two speakers based on the history. Here's the persona of speaker A to guide the conversation." 
instruction_one_persona_b = "Generate the next utterence in the interesting conversation between two speakers based on the history. Here's the persona of speaker B to guide the conversation."
instruction_both_persona = "Generate the next utterence in the interesting conversation between two speakers based on the history. Here's the persona of speaker A and B to guide the conversation."
instruction_no_persona = "Generate the next utterence in the interesting conversation between two speakers based on the history."  

def get_history_string(history,history_limit=15):     
    history_labeled = []     
    for j, utterance in enumerate(history):         
        if j % 2 == 0:             
            history_labeled.append("A: " + utterance)         
        else:             
            history_labeled.append("B: " + utterance)     
        history = history_labeled[-history_limit:]     
        #concatenate the history spaces, there should be no leading space     
    return "\n".join(history)  

def create_input_one_speaker(personality, history, person):     
    return "### Speaker "+person+" Persona: " + personality + "\n### Conversation History: " + history

def create_input_both_speaker(personality_a, personality_b, history):     
    return "### Speaker A Persona: " + personality_a + "\n### Speaker B Persona: " + personality_b + "\n### Conversation History: " + history  

def create_input_no_speaker(history):     
    return "### Conversation History: " + history 

def parse_conversation(conversation):
    parsed = []
    lines = conversation.split(' ')
    speaker = ''
    utterance = ''
    for line in lines:
        if line.endswith(':'):
            if speaker != '' and utterance != '':
                parsed.append({'speaker': speaker, 'utterance': utterance.strip()})
                speaker = ''
                utterance = ''
            speaker = line[:-1]
        else:
            utterance += line + ' '
    if speaker != '' and utterance != '':
        parsed.append({'speaker': speaker, 'utterance': utterance.strip()})

    # if consecutive speakers are the same, merge their utterances
    merged = []
    for i, utterance in enumerate(parsed):
        if i == 0:
            merged.append(utterance)
        elif utterance['speaker'] == merged[-1]['speaker']:
            merged[-1]['utterance'] += ' ' + utterance['utterance']
        else:
            merged.append(utterance)
    return merged

def get_entries_from_conversation(entry,persona_perm=1, history_limit=3):     
    entries = []
    persona_a = entry['Person A']
    persona_b = entry['Person B']
    conversation_split = parse_conversation(entry['conversation'])
    history = []
    for utterence in conversation_split:
        mode =  random.random()
        history_string = "\n".join([utter['speaker']+": "+utter['utterance'] for utter in history][-history_limit:])
        if mode < 0.25:                 
            instruction = instruction_no_persona                 
            input_string = create_input_no_speaker(history_string)             
        elif mode < 0.50:                          
            instruction = instruction_one_persona_a                 
            input_string = create_input_one_speaker(persona_a, history_string,"A")    
        elif mode < 0.75: 
            instruction = instruction_one_persona_b                
            input_string = create_input_one_speaker(persona_b, history_string,"B")    
        else:
            instruction = instruction_both_persona             
            input_string = create_input_both_speaker(persona_a,persona_b, history_string)    
        entries.append({                
            "mode": mode,
            "instruction": instruction,                 
            "input": input_string,                 
            "response": utterence['speaker']+": "+utterence['utterance'],             
        })
        history.append(utterence)
    return entries 

#function to download the dataset from the url if it is not already present 
def download_personachat_dataset():     
    if os.path.exists("personachat_self_original.json"):         
        return     
    r = requests.get(PERSONACHAT_URL)     
    with open("personachat_self_original.json", "w") as f:         
        f.write(r.text)  

#get the dataset, iterate over the entries and get the entries from the conversation 
def get_personachat_dataset():    
    with open("data_28k_chunk.json") as f:         
        data = json.load(f)    
    entries = []        
    for entry in data:         
        entries.extend(get_entries_from_conversation(entry))  
        #save as a json file called personachat.json, format the json file     
    print("new dataset size: ", len(entries))     
    with open("fisher_new.json", "w") as f:         
        json.dump(entries, f, indent=2)         
    #save the dataset as a json file     
    return entries

get_personachat_dataset()