import time
import streamlit as st
import numpy as np
import pandas as pd

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
    select = st.selectbox('Which type do you want to upgrade ?', ['Noun', 'Adjective','Adverb']) # return type: list
    submit_button = st.form_submit_button(label='Submit')

st.write('text input:', text_input)
st.write('Upgrade type:', select)
    
with st.spinner(text='In progress'):
    for i in range(100):
        # Process language model
        time.sleep(0.1)
    st.success('Done')

# output result
