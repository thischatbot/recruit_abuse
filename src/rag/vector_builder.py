from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from pathlib import Path
import os
from .constants import CATEGORY_LIST

BASE_DIR = Path(__file__).resolve()
while not (BASE_DIR / "docs").exists():
    BASE_DIR = BASE_DIR.parent
db_root_path = BASE_DIR / "vector_dbs"

def save_local_vector_db(api_key):
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = UpstageEmbeddings(api_key=api_key, model="solar-embedding-1-large-passage")
    
    for category, files in category_docs.items():
        save_path = os.path.join(db_root_path, category)
        docs = []
        for file in files:
            full_path = BASE_DIR / "docs" / file
            loader = PyPDFLoader(str(full_path))
            docs.extend(loader.load_and_split(splitter))
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(save_path)
        saved_paths.append(save_path)
        
    return saved_paths        