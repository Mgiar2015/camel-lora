import sys
import os
import random
import torch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter

load_8bit: bool = False
base_model: str = "decapoda-research/llama-7b-hf"

prompt_template: str = "alpaca"  # The prompt template to use, will default to alpaca
persona_list = [
    "This person is an international student who recently came to a new place and does not have much time to watch TV. They see TV as limited in choice and mostly unappealing. Their goal in this conversation is to find common ground with Person B who is discussing different TV shows.",
    "This person is a college student who enjoys celebrating Thanksgiving, but wishes there were more holidays that focused on thanksgiving and being together, rather than getting gifts. They also have experience with different religious holidays and how schools accommodate for them.",
    "This person is a fitness enthusiast who loves trying new workouts and sharing their experiences with others. They enjoy helping others achieve their fitness goals and find motivation in their progress.",
    "This person is a software engineer who enjoys coding, learning new programming languages, and staying up-to-date with technology trends. They are passionate about problem-solving and applying technology to improve people's lives.",
    "This person is a passionate environmentalist who is actively involved in conservation efforts and promoting eco-friendly practices. They are knowledgeable about climate change and its effects and are eager to educate others on how to reduce their carbon footprint.",
    "This person is a professional musician who plays multiple instruments and has experience performing in various venues. They are passionate about music and love discussing different genres, artists, and techniques with others.",
    "This person is a stay-at-home parent who manages the household and takes care of their children. They enjoy cooking, organizing family activities, and volunteering at their children's school. They are always looking for ways to improve their parenting skills and create a nurturing environment.",
    "This person is an avid traveler who has visited numerous countries and experienced different cultures. They enjoy sharing their travel experiences and advice with others and are always planning their next adventure.",
    "This person is a history buff who loves reading about historical events and visiting museums. They have a vast knowledge of different time periods and enjoy engaging in conversations about history and its impact on the present.",
    "This person is a freelance graphic designer who enjoys creating visually appealing designs for clients. They have a keen eye for aesthetics and stay updated on the latest design trends. They are passionate about helping businesses and individuals communicate their ideas effectively through visuals.",
    "This person is a foodie who enjoys exploring new restaurants and trying different cuisines. They love sharing their culinary experiences with friends and family and are always on the lookout for the next hidden gem.",
    "This person is a sports fan who follows multiple sports and enjoys attending live events. They have a deep understanding of game strategies and player statistics and love discussing the latest sports news with other enthusiasts.",
    "This person is a high school teacher who is passionate about education and helping students achieve their full potential. They are always looking for new teaching strategies and resources to engage their students and make learning enjoyable.",
    "This person is a retired senior who enjoys spending time with their grandchildren, gardening, and participating in community events. They have a wealth of life experience and enjoy sharing their wisdom with others.",
    "This person is a young professional working in the finance industry. They enjoy analyzing data, making informed decisions, and helping others with their financial goals. They are always looking to expand their knowledge and grow their network.",
    "This person is an aspiring entrepreneur who is in the process of starting their own business. They are eager to learn from successful entrepreneurs and are passionate about turning their ideas into reality.",
    "This person is a movie enthusiast who loves watching and discussing films from various genres. They enjoy analyzing film techniques, storytelling, and cinematography and are always excited to share their opinions and recommendations.",
    "This person is a pet owner who loves spending time with their furry friends and learning about animal behavior and care. They are knowledgeable about pet health, training, and proper care and enjoy sharing tips and advice with other pet owners.",
    "This person is an amateur photographer who enjoys capturing moments and experimenting with different photography techniques. They have a keen eye for composition and are passionate about sharing their work with others and learning from fellow photography enthusiasts.",
    "This person is an avid reader who loves exploring different genres and discussing books with others. They are part of a book club and enjoy attending author events and book signings. They are always eager to share their latest book recommendations and discuss themes and characters."]
list_of_topics = [
    "traveling alone",
    "healthy eating habits",
    "online learning platforms",
    "work-life balance",
    "urban gardening",
    "virtual reality gaming",
    "international cuisine",
    "smart home technology",
    "mental health and self-care",
    "exercise and fitness routines",
    "exploring national parks",
    "freelance work and remote jobs",
    "public transportation systems",
    "wildlife conservation",
    "music festivals and concerts",
    "baking and pastry arts"
]
temperature: float = 0.8
top_p: float = 0.6
top_k: int = 40
num_beams: int = 2
max_new_tokens: int = 150

prompter = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)

def generate_conversation(convo_num, model, lora_weights, temperature, top_p):
    p_a = random.choice(persona_list)
    p_b = random.choice(persona_list)

    while p_a == p_b:
        p_b = random.choice(persona_list)

    print(f"Person A: {p_a}")
    print(f"Person B: {p_b}")
    # conversation_history = input("Give the history (optional), e.g. A: Hey how are you doing? B: I'm great, how are you?")
    topic = random.choice(list_of_topics)
    conversation_history = f"A: Hey, let's talk about ourselves" #if (conversation_history is None or len(conversation_history) < 4) else conversation_history
    print(f"A: Hey, let's talk about ourselves" if (conversation_history is None or len(conversation_history) < 4) else conversation_history)
    prompt = f"Generate the next utterance in the interesting and diverse conversation between two speakers based on the history. Here's the persona of speaker A and B to guide the conversation. Person A: {p_a}, Persona B: {p_b}.\nConversation History:\n{conversation_history}\n"
    
    temp = temperature
    topp = top_p
    
    for i in range(10):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temp,
            top_p=topp,
            top_k=top_k,
            num_beams=num_beams,
        )

        temp = 0.8
        topp = 0.6

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=int(len(prompt)*0.7) + max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        if output.startswith("<unk>"+prompt):
            output = output[len(prompt):]

        import re
        # breakpoint()
        pattern = r'^(A|B):.*$'
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            utterance = match.group()
            if len(utterance) < 200 and str(utterance[3:]+'\n') not in prompt:
                prompt += f"{utterance}\n"
                print(utterance)
            else:
                # generate a random number between 0 and 1 (inclusive) 1 d.p.
                temp       = float(random.randint(0, 10)/10)
                topp       = float(random.randint(0, 10)/10)
        else:
            breakpoint()
            print("doesn't start with the prompt")
        
        # save the conversation
    with open(f"convos/conversation_{lora_weights}_{convo_num}_special_modprompt.txt", "a") as f:
        f.write(prompt)

def run_exp(lora_weights):
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print('\n\n\n\n')
    for i in range(5):
        print(f"============ CONVO {i} - weights: {lora_weights} - special, mod prompt ================")
        generate_conversation(i, model, lora_weights, temperature, top_p)
        print(f"=======================================")


lw = ["camel-alpaca-fisher-v2", "camel-fisher-v2", "camel-alpaca-personachat", "camel-personachat-v2"]
for lora_weights in lw:
    run_exp(lora_weights)
