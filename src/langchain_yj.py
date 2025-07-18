from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
import streamlit as st
import os

category_list = ["gender", "age", "abuse"]
#path for saving vector DB
db_root_path = os.getcwd() + "/../vector_dbs"

@st.cache_data
def save_local_vector_db(api_key):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    # prepare for docs
    category_docs = {
        "gender": [
            "gender/남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률(법률)(제20521호)(20250223).pdf"
        ],
        "age": [
            "age/고용상 연령차별금지 및 고령자고용촉진에 관한 법률(법률)(제18921호)(20220610).pdf"
        ],
        "abuse": [
            "abuse/근로기준법_직장_내_괴롭힘.pdf",
            "abuse/채용절차의 공정화에 관한 법률(법률)(제17326호)(20200526).pdf"
        ]
    }

    saved_paths = []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    embeddings = UpstageEmbeddings(
        api_key=api_key,
        model="solar-embedding-1-large-passage"
    )
    for category, files in category_docs.items():
        save_path = os.path.join(db_root_path, category)
        if os.path.exists(save_path):
            pass
        
        file_docs = []
        for file in files:
            path = os.getcwd() + "/../docs/" + file
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                file_docs.extend(loader.load())

        split_documents = splitter.split_documents(file_docs)
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore.save_local(save_path)
        saved_paths.append(save_path)
    
    return saved_paths
        
def load_local_vector_db (api_key) -> FAISS :
    vectorstore_list = []
    
    embeddings = UpstageEmbeddings(
        api_key=api_key,
        model="solar-embedding-1-large-passage"
    )
    
    vectordb_merged = None
    for category in category_list:
        save_path = os.path.join(db_root_path, category)
        # allow_dangerous_deserialization should be True ONLY if all the vectorstore files can be trusted
        if os.path.exists(save_path):
            vectorstore = FAISS.load_local(save_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            if vectordb_merged is not None:
                vectorstore.merge_from(vectordb_merged)
            vectordb_merged = vectorstore
            
    return vectordb_merged

def analyze_interview(user_input: str, api_key: str, vectorstore) -> str:
    from langchain_upstage import ChatUpstage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.prompts import PromptTemplate
    
    chat = ChatUpstage(api_key=api_key)
    # labeling 
    summarization_prompt = PromptTemplate.from_template(
        """
        다음 면접 후기를 객관적 상황으로 요약하세요.
        "성차별", "나이차별", "모욕적 언행" 중 하나라도 해당되는 상황이 있으면 생략하지 마세요.
        
        면접 후기 : {text}
        
        성차별: 여성/남성 선호, 외모 평가, 출산/결혼 관련 질문 등을 이야기 합니다.
        나이 차별: “젊은 인재”, “30대 이하 우대” 등 법적으로 문제 있는 표현을 이야기 합니다.
        모욕적 언행: 지원자를 하대하는 표현, 부적절한 반말, 강압적 어투 그리고 지역 차별, 병역 등 민감한 요소 언급을 하거나 혐오 발언이 포함된 표현을 이야기 합니다.
        """
    )
    summarization_chain = ({"text": RunnablePassthrough()} | summarization_prompt | chat | StrOutputParser())
    summarized_output = summarization_chain.invoke(user_input)
    print("요약된 문장 : " + summarized_output)
    
    law_docs = []
    
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7},
    )
    law_docs.extend(retriever.invoke(summarized_output))

    if not law_docs:
        context = "관련 법률 문서가 없습니다."
    else:
        context = "\n\n".join([doc.page_content for doc in law_docs]) 

    law_explain_prompt = PromptTemplate.from_template(
        """
        당신은 법률 전문가입니다.
        다음은 면접 중 발화된 문장입니다.
        
        문장: {text}
        분류:
        설명:

        알맞은 법률 조항을 언급하여 다음 문장이 어떤 법, 조항에 위반되는지 분류하고 설명을 하세요.
        객관적으로 법을 참고하여 "성차별", "연령차별", "모욕적 언행" 중에서 분류를 하세요.
        법에 위반되지 않으면 분류하지 마세요.
        모욕적 언행은 직장 내 괴롭힘 금지법을 참고하세요.
        두개 이상 모두 해당되면 모두 포함시키세요.
        
        [관련 법]
        {context}
        """
    )

    law_explain_chain = ({"context": RunnableLambda(lambda x: context), "text": RunnablePassthrough()} 
                        | law_explain_prompt | chat | StrOutputParser())
    law_explanation = law_explain_chain.invoke(user_input)
    return law_explanation