import streamlit as st
from dotenv import load_dotenv
import os
from langchain_yj import analyze_interview

#load enviroment variables
load_dotenv()
api_key = os.getenv("SOLAR_API_KEY")

st.set_page_config(page_title="갑질 발언 분석기", layout="centered")

#initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = ["", "", ""]

questions = [
    "1. 면접에서 채용까지의 과정을 적어주세요",
    "2. 면접 중 기억에 남는 질문은 없었나요?",
    "3. 면접 중 기분 나빴던 발언은 있었나요?",
]

questions_summary = [
    "1. 면접부터 채용까지의 과정 : ",
    "\n\n 2. 기억에 남는 질문 : ",
    "\n\n 3. 기분 나빴던 발언 : ",
]

def next_step():
    if st.session_state.answers[st.session_state.step].strip() != "":
        st.session_state.step += 1
        
st.title("갑질 job아드립니다")
st.markdown("### 기분 job치는 갑질 면접, 법으로 job다")

#sliding card
if st.session_state.step < len(questions):
    st.markdown(f"### {questions[st.session_state.step]}")
    #input text
    st.text_area(
        label="",
        key=f"answer_{st.session_state.step}",
        value=st.session_state.answers[st.session_state.step],
        on_change=lambda: st.session_state.answers.__setitem__(st.session_state.step, st.session_state.get(f"answer_{st.session_state.step}")),
        height=150,
    )
    st.button("다음", on_click=next_step)
else:
    user_input_list = []
    #start analyze
    for question, answer in zip(questions_summary, st.session_state.answers):
        user_input_list.append(question)
        user_input_list.append(answer)
    user_input = "\n".join(user_input_list)
    st.markdown("### 입력한 내용")
    st.markdown(user_input)
    st.markdown("### 분석한 내용")
    st.markdown(analyze_interview(user_input=user_input, api_key=api_key))