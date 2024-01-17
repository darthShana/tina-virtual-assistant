from itertools import zip_longest
from backend.tina import Tina
import streamlit as st
from streamlit_chat import message

st.header("Turners Virtual Consultant")


if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "sales_controller" not in st.session_state:
    st.session_state["sales_controller"] = Tina()
    st.session_state["sales_controller"].seed_agent()
    st.session_state["chat_answer_history"] = []
    st.session_state["chat_answer_history"].append(st.session_state["sales_controller"].last_ai_message())


def submit():
    with st.spinner("lets have a look..."):
        st.session_state["sales_controller"].answer(st.session_state.prompt)
        st.session_state["user_prompt_history"].append(st.session_state.prompt)
        st.session_state["chat_answer_history"].append(st.session_state["sales_controller"].last_ai_message())
        st.session_state.prompt = ''


if st.session_state["chat_answer_history"]:
    for generated_response, user_query in zip_longest(st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]):
        if generated_response:
            message(generated_response)
        if user_query:
            message(user_query, is_user=True)


prompt = st.text_input("Prompt", key='prompt', on_change=submit, placeholder="your text here...")



