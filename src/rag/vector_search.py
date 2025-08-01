from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from .constants import CATEGORY_LIST
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
db_root_path = BASE_DIR / "vector_dbs"

def load_vector_db(category, api_key):
    embeddings = UpstageEmbeddings(
        api_key=api_key,
        model="solar-embedding-1-large-paasage"
    )
    db_path = os.path.join(db_root_path, category)
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def search_similar_docs(query, category, api_key, k=3):
    db = load_vector_db(category, api_key)
    docs = db.similarity_search(query, k=k)
    return docs