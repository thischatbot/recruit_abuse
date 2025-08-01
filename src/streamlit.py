import streamlit as st
from dotenv import load_dotenv
import os

from rag.vector_builder import save_local_vector_db
from rag.query_chain import run_legal_rag

#load enviroment variables
load_dotenv()
api_key = os.getenv("SOLAR_API_KEY") or st.secrets["SOLAR_API_KEY"]

st.set_page_config(page_title="갑질 발언 분석기", layout="centered")

# upper: add button
if st.button("📦 벡터 DB 저장 (최초 1회만 누르세요!)"):
    with st.spinner("문서를 분석하고 벡터 DB를 저장 중입니다..."):
        saved_paths = save_local_vector_db(api_key)
    st.success(f"다음 경로에 저장 완료됨:\n\n" + "\\n".join(saved_paths))

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
st.markdown("### 기분 job치는 갑질 면접💥 법으로 job다💥")

#sliding card
if st.session_state.step < len(questions):
    st.markdown(f"### {questions[st.session_state.step]}")
    st.text_area(
        label="",
        key=f"answer_{st.session_state.step}",
        value=st.session_state.answers[st.session_state.step],
        on_change=lambda: st.session_state.answers.__setitem__
        (st.session_state.step,
         st.session_state.get(f"answer_{st.session_state.step}")
         ),
        height=150,
    )
    st.button("✅ 다음", on_click=next_step)
else:
    user_input = "\n".join([
        q + "\n" + a for q, a in zip(questions_summary, st.session_state.answers)
    ])
    
    st.markdown("### 🗣 입력한 내용")
    st.write(user_input)
    
    st.markdown("### 🔍 분석한 내용")
    with st.spinner("분석 중입니다..."):
        category = "abuse"
        result = run_legal_rag(api_key=api_key, user_question=user_input, category=category)
    st.write(result)