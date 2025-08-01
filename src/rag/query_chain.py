from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from .vector_search import search_similar_docs
from .prompt_templates import LAW_EXPLAIN_TEMPLATE

def summarize_question(api_key, user_question):
    llm = ChatUpstage(
        api_key=api_key,
        model="solar-1-mini-chat",
        temperature=0.3,
        max_tokens=300   
    )
    prompt = PromptTemplate(
        input_variables=["question"],
        template="질문을 핵심 키워드로 요약해줘:\n\n{question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(user_question)

def make_rag_chain(api_key, category):
    llm = ChatUpstage(
        api_key=api_key,
        model="solar-1-mini-chat",
        temperature=0.3,
        max_tokens=512
    )
    rag_prompt = ChatPromptTemplate.from_template(LAW_EXPLAIN_TEMPLATE)
    return LLMChain(llm=llm, prompt=rag_prompt)

def run_legal_rag(api_key, user_question, category):
    summary = summarize_question(api_key, user_question)
    docs = search_similar_docs(summary, category, api_key)
    context = "\n".join([doc.page_content for doc in docs])
    rag_chain = make_rag_chain(api_key, category)
    return rag_chain.run({"context": context, "question": user_question})