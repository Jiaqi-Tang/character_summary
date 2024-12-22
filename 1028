import spacy
import coreferee
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import json

'''
input text (str): A story book
len_threshold (int): Up to what size (in characters) to regroup the paragraphs
divider (str): sperator of paragraphs
return (list of str): Regrouped paragraphs

Used to group smaller paragraphs (i.e. dialogs) to gether so there is more context 
for coreference resolving and summarizing
'''
def regroup_paragraphs(input_text, len_threshold, divider = '\n\n'):
    paras = input_text.split(divider)

    processed_paras = []
    temp = ''

    for para in paras:
        if len(para) > len_threshold:
            if temp:
                processed_paras.append(temp)
                temp = ''
            processed_paras.append(para)
        else:
            if temp:
                temp += ' ' + para
                if len(temp) > len_threshold:
                    processed_paras.append(temp)
                    temp = ''
            else:
                temp_para = para

    if temp_para:
        processed_paras.append(temp_para)
        
    return processed_paras


'''
para (str): a paragraph of the story book
doc (Doc): a spacy doc with coreferee pipline added
return (list of [int, int, str]): indicates which part of the string to replace with what

Replaces indirect mentions with the most specfic mention
'''
def resolve_chains(para, doc):
    to_replace = []
    if '"' not in para: # Coreference does not work well with dialogs
        entity_texts = [ent.text for ent in doc.ents]
        for chain in doc._.coref_chains:
            direct_mention = doc[chain[chain.most_specific_mention_index].root_index].text
            if direct_mention in entity_texts:
                for mention in chain:
                    index = mention.root_index
                    if len(mention.token_indexes) <= 1:
                        to_replace.append([doc[index].idx, len(doc[index].text), direct_mention])
    return to_replace


'''
para (str): paragraph of a stroy book
to_replace (list of [int, int, str]): indicates which part of the string to replace with what
return (str): paragraph with mentions resolved

Helper function for replacing indirect mentions with the most specfic mention
'''
def resolve_paragraph(para, to_replace):
    temp = para
    to_replace.sort(key=lambda x: x[0])
    offset = 0

    for start, length, text in to_replace:
        temp = temp[:start+offset] + text + temp[start+length+offset:]
        offset = offset + len(text) - length

    return temp
    

'''
nums (list of int): ordered list of int
return (list of list of int): input ints split into lists where all ints in each list 
are consecutive or differ by 1
'''
def split_consecutive(nums):
    if not nums:
        return []
    
    result = [[nums[0]]]
    
    for i in range(1, len(nums)):
        # Check if the current number is consecutive to the previous one
        if nums[i] <= nums[i - 1] + 2:
            result[-1].append(nums[i])
        else:
            result.append([nums[i]])
    
    return result

'''
text (str): string of text
return (list of str): chunked text

Helper function for chunking long text so they meet the requirements of the model
'''
def chunk_text(text, max_tokens=1024, overlap=200):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(chunk)
    return chunks


'''
paras (list of str): paragraphs of a story book
return (list of str, dict of [str, list of int]): paragraphs with coreferences resolved, dictionary keeping 
track of the paragraphs where each character appeared

For the preprocessing of paragraphs. Resolves coreference and makes note of which characters 
appear in which paragraphs
'''
def parse_paragraphs(paras):
    resolved = []
    char_paras = {}
    
    for para_num, para in enumerate(paras):
        # Resolves coreference
        doc = nlp(para)
        to_replace = resolve_chains(para, doc)
        resolved_para = resolve_paragraph(para, to_replace)

        # makes note of which characters appear in which paragraphs
        unique_ent = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
        for ent_text in unique_ent:
            if ent_text in char_paras:
                char_paras[ent_text].append(para_num)
            else:
                char_paras[ent_text] = [para_num]
    
        resolved.append(resolved_para)   
    return resolved, char_paras


'''
text (str): summarizes text
model (BartForConditionalGeneration): Bart model used to generate summaries
tokenizer (BartTokenizer): Bart tokenizer to tokenize text
return (str): summarized text
'''
def summarize_text(text, model, tokenizer):
    chunks = chunk_text(text) # Chunks text to overlap, thus giving more context to the modal
    summaries = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk])
        summary_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)


'''
chars (list of str): all characters to generate summaries for
char_paras (dict of [str, list of int]): dictionary keeping track of the paragraphs where each character appeared
resolved_paras (list of str): paragraphs with coreferences resolved
return (dict of [str, list of str]): summaries for each character

For generating summaries for each character
'''
def generate_summaries(chars, char_paras, resolved_paras):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    char_summaries = {}
    
    for char in chars:
        summary_list = []
        para_lists = split_consecutive(char_paras[char]) # Groups consecutive paragraphs together for more context
        
        for para_list in para_lists:
            if not para_list:
                continue
            paras = [resolved_paras[para_num] for para_num in range(para_list[0], para_list[-1] + 1)]
            text = " ".join(paras)
            summary_list.append(summarize_text(text, model, tokenizer))
        
        char_summaries[char] = summary_list
    return char_summaries


'''
char_sums (dict of [str, list of str]): summaries for each character

prints all character summaries
'''
def print_characters(char_sums):
    for character, summaries in char_sums.items():
        print("Character: " + character)
        for i in range(len(summaries)):
            summaries[i] = re.sub(r'\s{2,}', ' ', summaries[i])
            print(summaries[i])
        print("")

STORY_FILE = "harry_potter.txt"
EXPORT_FILE = "character_summaries.json"

# Loads story file
with open(STORY_FILE, "r") as f:
    text = f.read()
    
paragraphs = regroup_paragraphs(text, 400)

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe('coreferee')

resolved_paragraphs, character_paragraphs = parse_paragraphs(paragraphs)

### placeholder for merging first/lastnames together

characters = [key for key, value in character_paragraphs.items() if len(value) >= 10]

character_summaries = generate_summaries(characters, character_paragraphs, resolved_paragraphs)
print_characters(character_summaries)

# Exports the dictionary to a JSON file
with open(EXPORT_FILE, "w") as json_file:
    json.dump(character_summaries, json_file, indent=4) 


