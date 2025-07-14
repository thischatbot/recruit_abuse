from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

retriever_dict = {}

def get_retreiver_by_label(label: str):
    if "성차별" in label:
        return retriever_dict["gender"]
    elif "나이차별" in label:
        return retriever_dict["age"]
    elif "모욕적 언행" in label:
        return retriever_dict["abuse"]
    else:
        return None

#load enviroment variables
load_dotenv()
api_key = os.getenv("SOLAR_API_KEY")
chat = ChatUpstage(api_key=api_key)

user_input = "여자는 연애를 하고 결혼을 하고 애를 낳아야 한다. 그렇지 않으면 같이 일 못하겠다."

# labeling 
classification_prompt = PromptTemplate.from_template(
    """
    당신은 법률 전문가입니다.
    다음은 면접 중 발화된 문장과 그에 대한 분류 예시입니다.
    
    문장: (여성 지원자에게) 육아휴직 쓸 텐데, 뽑지 말자
    분류: 성차별

    문장: 여자는 애 낳아야지.
    분류: 성차별

    문장: 여성은 연애 → 결혼 → 출산을 해야 한다. 그렇게 안 하는 사람을 우리가 어떻게 믿고 같이 일하냐
    분류: 성차별

    문장: 여자들은 대부분 결혼하면 일을 그만둬서 여자를 뽑기 꺼려진다. 다들 결혼하면 그만둘 거냐?
    분류: 성차별

    문장: 65세 이상은 다 내보내라
    분류: 나이 차별

    문장: 나이가 몇인데 경력이 이거밖에 안 돼
    분류: 나이 차별

    문장: (암 투병 중인 가족을 간호하느라 공백이 생긴 사람에게) 직장보다 가족이 우선인 사람은 싫다.
    분류: 나이 차별

    문장: 부모님이 언제 이혼하셨나? 그래서 성격에 문제가 있는 거 아니야?
    분류: 모욕적 언행

    문장: 정치 성향이 진보인지, 보수인지 답변해달라.
    분류: 모욕적 언행

    문장: 웃겨 보라.
    분류: 모욕적 언행

    이제 다음 문장을 분류해 주세요.

    문장: {text}
    분류:

    **참고사항**
    성차별: 여성/남성 선호, 외모 평가, 출산/결혼 관련 질문 등을 이야기 합니다.
    모욕적 언행: 지원자를 하대하는 표현, 부적절한 반말, 강압적 어투 그리고 지역 차별, 병역 등 민감한 요소 언급을 하거나 혐오 발언이 포함된 표현을 이야기 합니다.

    [성차별], [나이차별], [모욕적 언행] 세가지 카테고리 중 포함되는 내용으로 분류하세요
    세가지 중 아무것도 포함되지 않으면 [판별 불가] 라고 출력하세요.
    세가지 중 포함되는 대상이 2개 이상이면 해당 카테고리를 모두 출력하세요.
    """
)
classification_chain = ({"text": RunnablePassthrough()} | classification_prompt | chat | StrOutputParser())
classification_output = classification_chain.invoke(user_input)
#print(classification_output)
label = classification_output.split(":")[1].split('\n')[0].strip()
print(label)
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

for category, files in category_docs.items():
    docs = []
    for file in files:
        path = os.getcwd() + "/../docs/" + file
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

    split_documents = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    save_path = os.path.join(db_root_path, category)
    vectorstore.save_local(save_path)
    
    retriever_dict[category] = vectorstore.as_retriever()

retriever = get_retreiver_by_label(label)
if retriever is None:
    print("retriever 호출 실패")

law_explain_prompt = PromptTemplate.from_template(
    """
    당신은 법률 전문가입니다.
    다음은 면접 중 발화된 문장과 그에 대한 위반 법 분류입니다.
    
    문장: {text}
    분류: {law_classification}

    알맞은 법률 조항을 언급하여 다음 문장이 법에 위반되는지 설명을 하세요.
    
    [성차별 관련 법]
    {context}
    """
)

law_explain_chain = ({"context": retriever, "text": RunnablePassthrough(), "law_classification": RunnableLambda(lambda x: classification_output)} 
                     | law_explain_prompt | chat | StrOutputParser())
law_explanation = law_explain_chain.invoke(user_input)
print(law_explanation)