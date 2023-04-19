
import json
import requests 
import os 
import random  
PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"  
instruction_one_persona = "Generate the next utterence in the interesting and casual conversation between two speakers based on the provided conversation history. Some information on the persona of speaker B is provided below and can help guide the conversation." instruction_no_persona = "Generate the next utterence in the interesting and casual conversation between two speakers based on the provided conversation history."  

def get_history_string(history,history_limit=20):     
    history_labeled = []     
    for j, utterance in enumerate(history):         
        if j % 2 == 0:             
            history_labeled.append("A: " + utterance)         
        else:             
            history_labeled.append("B: " + utterance)     
        history = history_labeled[-history_limit:]     
        #concatenate the history spaces, there should be no leading space     
    return  " ".join(history)  

def create_input_one_speaker(personality, history):     
    return "### Speaker B Persona: " + personality + "\n### Conversation History: " + history  

def create_input_no_speaker(history):     
    return "### Conversation History: " + history  

def get_entries_from_conversation(entry,persona_perm=1, history_limit=3):     
    entries = []    
    for i in range(persona_perm):         
        for utterance in entry["utterances"]:              
            candidates = utterance["candidates"]             
            history_string = get_history_string(utterance["history"])             
            persona = " ".join(entry["personality"])             
            if random.random() < 0.15:                 
                instruction = instruction_no_persona                 
                input_string = create_input_no_speaker(history_string)             
            else:                 
                instruction = instruction_one_persona                 
                input_string = create_input_one_speaker(persona, history_string)             
            entries.append({                 
                "instruction": instruction,                 
                "input": input_string,                 
                "response": "B: "+candidates[-1],             
            })         
            #persona = [persona[-1]] + persona[:-1]     
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
    download_personachat_dataset()     
    with open("personachat_self_original.json") as f:         
        data = json.load(f)    
    entries = []    
    print("dataset size: ", len(data["train"]))     
    for entry in data["train"]:         
        entries.extend(get_entries_from_conversation(entry))     
        #save as a json file called personachat.json, format the json file     
    print("new dataset size: ", len(entries))     
    with open("personachat_new.json", "w") as f:         
        json.dump(entries, f, indent=2)         
    #save the dataset as a json file     
    return entries

get_personachat_dataset()