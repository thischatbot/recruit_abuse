from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

#load enviroment variables
load_dotenv()
api_key = os.getenv("SOLAR_API_KEY")
chat = ChatUpstage(api_key=api_key)

user_input = input()

# RAG

# create vector DB
loader = PyPDFLoader(os.getcwd() + "/../docs/gender/gender_hiring_law.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
split_documents = splitter.split_documents(docs)
print(len(split_documents))
# do embedding
embeddings = UpstageEmbeddings(
    api_key=api_key,
    model="solar-embedding-1-large-query"
)

vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
retriever = vectorstore.as_retriever()

#TODO: 분류만 하는 프롬프트로 바꾸기
prompt = PromptTemplate.from_template(
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
    설명:

    **참고사항**
    성차별: 여성/남성 선호, 외모 평가, 출산/결혼 관련 질문 등을 이야기 합니다.
    모욕적 언행: 지원자를 하대하는 표현, 부적절한 반말, 강압적 어투 그리고 지역 차별, 병역 등 민감한 요소 언급을 하거나 혐오 발언이 포함된 표현을 이야기 합니다.

    [성차별], [나이차별], [모욕적 언행] 세가지 카테고리 중 포함되는 내용으로 분류하세요
    세가지 중 아무것도 포함되지 않으면 [판별 불가] 라고 출력하세요.
    세가지 중 포함되는 대상이 2개 이상이면 해당 카테고리를 모두 출력하세요.
    다음 법률 조항을 언급하여 설명을 하세요.
    
    [성차별 관련 법]
    {context}
    """
)

chain = ({"context": retriever, "text": RunnablePassthrough()} | prompt | chat | StrOutputParser())
response = chain.invoke(user_input)
print(response)

# label = response.content.split(":")[1].split('\n')[0].strip()
# comment = response.content.split("설명:")[1].strip()