import pandas as pd
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# load enviroment variables
load_dotenv()
api_key = os.getenv("SOLAR_API_KEY")
chat = ChatUpstage(api_key=api_key)

# load testset (.csv file)
df = pd.read_csv("testset.csv") # it needs text, label columm
results = []

for i, row in df.iterrows():
    prompt = f"""
    다음은 면접 중 발화된 문장과 그에 대한 분류 예시입니다.

    문장: (여성 지원자에게) 육아휴직 쓸 텐데, 뽑지 말자
    분류: 성차별

    문장: 여자는 애 낳아야지.
    분류: 성차별

    문장: 여성은 연애 → 결혼 → 출산을 해야 한다. 그렇게 안 하는 사람을 우리가 어떻게 믿고 같이 일하냐
    분류: 성차별

    문장: 여자들은 대부분 결혼하면 일을 그만둬서 여자를 뽑기 꺼려진다. 다들 결혼하면 그만둘 거냐?
    분류: 성차별

    문장: (항암치료로 머리가 짧은 여성에게) 남자냐, 여자냐?
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

    문장: 게슴츠레한 눈빛 때문에 태도가 불량해 보인다.
    분류: 모욕적 언행

    문장: 웃겨 보라.
    분류: 모욕적 언행

    문장: (면접자의 출신 대학을 보고) 이 대학 나와서 성실하다고 할 수 있나?
    분류: 모욕적 언행

    이제 다음 문장을 분류해 주세요.
    
    문장: {row['text']}
    분류:
    설명:

    **참고사항**
    성차별: 여성/남성 선호, 외모 평가, 출산/결혼 관련 질문 등을 이야기 합니다.
    모욕적 언행: 지원자를 하대하는 표현, 부적절한 반말, 강압적 어투 그리고 지역 차별, 병역 등 민감한 요소 언급을 하거나 혐오 발언이 포함된 표현을 이야기 합니다.

    [성차별], [나이차별], [모욕적 언행] 세가지 카테고리 중 포함되는 내용으로 분류하세요
    세가지 중 아무것도 포함되지 않으면 [판별 불가] 라고 출력하세요.
    세가지 중 포함되는 대상이 2개 이상이면 해당 카테고리를 모두 출력하세요.

    """

    messages = [
        SystemMessage(content="당신은 채용 갑질 발언을 분류하는 AI입니다. 성차별, 나이차별, 모욕적 언행 발언을 분류합니다."),
        HumanMessage(content=prompt)
    ]
    response = chat.invoke(messages)
    label = response.content.split(":")[1].split('\n')[0].strip()
    comment = response.content.split("설명:")[1].strip()
    results.append({
        "text" : row['text'],
        "pre_label" : row['label'],
        "gpt_label" : label,
        "comment" : comment
    })
    
pd.DataFrame(results).to_csv("gpt_testset_results.csv", index=False)
    