def analyze_interview(user_input: str, api_key: str) -> str:
    from langchain_upstage import ChatUpstage
    from langchain_upstage import UpstageEmbeddings
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.prompts import PromptTemplate
    import os
    
    chat = ChatUpstage(api_key=api_key)
    # labeling 
    summarization_prompt = PromptTemplate.from_template(
        """
        다음 문장을 요약하세요.
        "성차별", "나이차별", "모욕적 언행" 중 하나라도 해당되는 상황이 있으면 생략하지 마세요.
        "성차별", "나이차별", "모욕적 언행" 중 해당되지 않은 것이 있으면 포함시키지 마세요.
        주관적인 의견을 출력하지 마세요. 분류된 라벨을 출력하지 마세요.
        문장 : {text}
        """
    )
    summarization_chain = ({"text": RunnablePassthrough()} | summarization_prompt | chat | StrOutputParser())
    summarized_output = summarization_chain.invoke(user_input)
    print("요약된 문장 : " + summarized_output)
    #####
    # create vector DB

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    embeddings = UpstageEmbeddings(
        api_key=api_key,
        model="solar-embedding-1-large-query"
    )

    #path for saving vector DB
    db_root_path = os.getcwd() + "/../vector_dbs"
    law_docs = []
    retriever_dict = {}
    for category, files in category_docs.items():
        file_docs = []
        for file in files:
            path = os.getcwd() + "/../docs/" + file
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                file_docs.extend(loader.load())

        split_documents = splitter.split_documents(file_docs)
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

        save_path = os.path.join(db_root_path, category)
        vectorstore.save_local(save_path)
        
        retriever_dict[category] = vectorstore.as_retriever()
        law_docs.extend(retriever_dict[category].invoke(summarized_output))
        
            
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

        알맞은 법률 조항을 언급하여 다음 문장이 어떤 법에 위반되는지 분류하고 설명을 하세요.
        객관적으로 법을 참고하여 "성차별", "연령차별", "모욕적 언행" 중에서 분류를 하세요.
        법에 위반되지 않으면 분류하지 마세요.
        두개 이상 모두 해당되면 모두 포함시키세요.
        
        [관련 법]
        {context}
        """
    )

    law_explain_chain = ({"context": RunnableLambda(lambda x: context), "text": RunnablePassthrough()} 
                        | law_explain_prompt | chat | StrOutputParser())
    law_explanation = law_explain_chain.invoke(user_input)
    return law_explanation