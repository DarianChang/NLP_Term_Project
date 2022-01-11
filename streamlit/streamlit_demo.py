import time
import streamlit as st
# from processing import *

import pandas as pd
from pandas import DataFrame as df
import nltk
from transformers import pipeline
unmasker = pipeline('fill-mask', model='distilbert-base-uncased',top_k=100)
# model= "bert-base-uncased"

def proper_adj(sent): # sentence must contain target word, for now!!!
    ## 標注詞性 將形容詞mask
    # JJ     adjective    'big'  形容詞
    # JJR    adjective, comparative 'bigger' （形容詞或副詞的）比較級形式
    # JJS    adjective, superlative 'biggest'  （形容詞或副詞的）最高階
    mask_index = []
    words = nltk.word_tokenize(sent)
    tags = set(['JJ', 'JJR', 'JJS'])
    words_w_pos_tags = [list(i) for i in nltk.pos_tag(words)]
    # print(pos_tags)
    words_mask_list = []
    for idx,word in enumerate(words_w_pos_tags):
        # print(word[0])
        if word[1] in tags:
            word[0] = "[MASK]"
            mask_index.append(idx)
            # print(word[1])
        words_mask_list.append(word[0])
        
    # print(words_w_pos_tags)

    mask_sent = " ".join(words_mask_list)
    # print(mask_sent)
    mask_result = df(unmasker(mask_sent))
    # mask_result["mask_index"] = ' '.join(str(mask_index))
    mask_result["mask_index"] = [mask_index for i in range(mask_result.shape[0])] 
    # print(mask_result.shape[0])

    return mask_result
    ## 對推薦的字作形容詞篩選
    
    
def compare_pos(sent1,sent2,mask_index = None): # sentence contain have same pos structure -> true -> bool
    words_1 = nltk.word_tokenize(sent1)
    words_2 = nltk.word_tokenize(sent2)
    # pos的第二個元素才是詞性
    words_w_pos_tags_1 = [list(i[1]) for i in nltk.pos_tag(words_1)]
    words_w_pos_tags_2 = [list(i[1]) for i in nltk.pos_tag(words_2)]
    # print(words_w_pos_tags_1,words_w_pos_tags_2)
    if mask_index == None:
        if words_w_pos_tags_1 == words_w_pos_tags_2: return True
        else: return False
    elif mask_index is not None: 
        for i in mask_index: 
            if words_w_pos_tags_1[i] != words_w_pos_tags_2[i]: return False
        return True

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

##
hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)


def nli_sents(premise, hypothesis):
    max_length = 256
    # premise = "Two women are embracing while holding to go packages."
    # hypothesis = "The men are fighting outside a deli."
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one
    return {"Ent": round(predicted_probability[0],2),"Neu": round(predicted_probability[1],2),"Con": round(predicted_probability[2],2)}

def nli_result_append(input_sent,mask_result,mask_result_sent,k,forward_nli_E,backward_nli_E):
    mask_result["forward_nli_E"] = [nli_sents(input_sent,i)["Ent"] for i in mask_result_sent]
    mask_result["forward_nli_N"] = [nli_sents(input_sent,i)["Neu"] for i in mask_result_sent]
    mask_result["forward_nli_C"] = [nli_sents(input_sent,i)["Con"] for i in mask_result_sent]
    mask_result["backward_nli_E"] = [nli_sents(i,input_sent)["Ent"] for i in mask_result_sent]
    mask_result["backward_nli_N"] = [nli_sents(i,input_sent)["Neu"] for i in mask_result_sent]
    mask_result["backward_nli_C"] = [nli_sents(i,input_sent)["Con"] for i in mask_result_sent]
    filter_f_nli_E = (mask_result["forward_nli_E"] >= forward_nli_E )
    filter_b_nli_E = (mask_result["backward_nli_E"] >= backward_nli_E )
    return mask_result[(filter_f_nli_E & filter_b_nli_E)][:k]

# unmasker = pipeline('fill-mask', model='bert-base-uncased',top_k=100)

def proper_adj_topk(input_sent,k,forward_nli_E=0.5,backward_nli_E=0.5) : # -> {"sent":"text","forward_nli_E":"number","backward_nli_E":"number"}  
    mask_result = proper_adj(input_sent)
    mask_result_sent = list(mask_result.loc[:,"sequence"])
    mask_index = mask_result["mask_index"][0]
    # compare_pos(input_sent,i)
    mask_result["same_pos"] = [compare_pos(input_sent,i,mask_index) for i in mask_result_sent]
    filter = (mask_result["same_pos"] == True)
    # 篩選同詞性 先不篩
    mask_result = mask_result[filter]
    # print(mask_result)
    mask_result_sent = list(mask_result.loc[:,"sequence"])
    mask_result = nli_result_append(input_sent,mask_result,mask_result_sent,k,forward_nli_E,backward_nli_E)
    # print(mask_result)
    return mask_result


st.set_page_config(
    page_title="NLP Demo",
    page_icon=":book",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title('2021 NLP Demo')

st.markdown("## Abstract")
st.markdown('Our Task is try to upgrade sentence. Base on CEFR, we can choose which type you want to upgrade.')
st.markdown('For Example,  input sentence: **An big accident.**')
st.markdown('Then, we try to upgrade adjective. (i.e. In this sentence, we\' re going to upgrade  "big\") ')
st.markdown("Output: **An serious accident**")
st.markdown("---")

st.markdown("## Demo")
with st.form(key='my_form'):
    text_input = st.text_input(label='Enter sentence')
    # noun = st.checkbox('Noun'),
    # adj = st.checkbox('Adjective'),
    # adv = st.checkbox('Adverb'),
    select = st.selectbox('Which type do you want to upgrade ?', ['Adjective','Noun','Adverb']) # return type: list
    submitted = st.form_submit_button(label='Submit')
    
    
    if submitted:
        st.success('submitted successfully')
        

st.write('text input:', text_input)
st.write('Upgrade type:', select)

# input and calculate the result
mask_result = proper_adj_topk(text_input,5)
mask_result_sequence_list = mask_result["sequence"]
# score	token	token_str	sequence	mask_index

with st.spinner(text='In progress'):
    for i in range(100):
        # Process language model
        time.sleep(0.1)
    st.success('Done')

# output result

# st.write('Output result:', select)

txt = st.text_area('Text to analyze', str(mask_result_sequence_list))

st.write('Output result:', txt)