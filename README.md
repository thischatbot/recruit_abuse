<img width="726" height="845" alt="image" src="https://github.com/user-attachments/assets/c1266172-94d2-4bee-92b1-2dfea97548cd" />

## 소개 (about)

**갑질 Job아드립니다**는 사용자로부터 면접 경험을 입력 받아 그 발언이 **성차별 / 나이차별 / 모욕적 언행**에 해당하는지 자동으로 분류하고, **관련 법률 조항을 기반으로 법적 해설**까지 제공하는 AI 기반 웹앱입니다

이 프로젝트는 LLM + RAG + 법률 데이터셋을 활용하여 다음과 같은 기능을 제공합니다

- 사용자 인터뷰 유도 UX (3단계 질문 카드)
- 사용자 입력 요약 및 분류
- RAG 기반 관련 법률 문서 검색
- 법률 전문가 스타일의 설명 생성


## 기술 스택

- **LangChain** (Chain 구성 및 Prompt Template)
- **Upstage SOLAR API** (`solar-embedding`, `chat`)
- **FAISS** (법률 문서 벡터 저장소)
- **Streamlit** (슬라이딩 카드형 UX 프론트엔드)
- **PyPDFLoader** (법률 PDF 로딩)
- **Python 3.11.8** 

---

## 실행 방법

1. 환경 설정
```bash
git clone https://github.com/thischatbot/recruit_abuse
cd recruit_abuse
pyenv virtualenv 3.11.8 job_abuse
pyenv activate job_abuse
pip install -r requirements.txt
```
2. .env 파일 설정
```
SOLAR_API_KEY=your_solar_api_key_here
```

3. 실행
```bash
streamlit run streamlit.py
```

---

## 📌 TODO

- [ ] 테스트 케이스 기반 정확도/응답 속도 보정 : 벡터 DB
- [ ] 테스트 케이스 기반 정확도 보정 : 프롬프트
- [ ] Streamlit Cloud 배포
