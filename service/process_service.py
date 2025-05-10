from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from langchain.schema import Document
import unicodedata
import os
from langchain_community.vectorstores import FAISS


from langchain_openai import ChatOpenAI

from langchain_core.runnables import Runnable

def load_pdf_pages(file: str):
    loader = PyPDFLoader(file)
    return loader.load()


def clean_text(text:str):
    """Remove espaços e quebras de linha redundantes."""
    return " ".join(text.split())

def extract_personal_and_experience_info(pages: List[Document]) -> Dict[str, str]:
    """
        Extrai informações pessoais e experiência profissional de um documento PDF processado.

        Args:
            pages (List[Document]): Lista de objetos Document contendo o conteúdo das páginas.

        Returns:
            Dict[str, str]: Um dicionário com as chaves 'informacoes_pessoais' e 'experiencia_profissional'.
        """
    # Junta todas as páginas em um único texto
    full_text = " ".join([page.page_content for page in pages])
    texto_limpo = clean_text(full_text)

    # Separa com base na palavra-chave 'EXPERIÊNCIA'
    partes = texto_limpo.split("EXPERIÊNCIA")
    if len(partes) >= 2:
        info_pessoal = partes[0].strip()
        experiencia = partes[1].split("FORMAÇÃO")[0].strip()
    else:
        info_pessoal = texto_limpo
        experiencia = ""

    return {
        "informacoes_pessoais": info_pessoal,
        "experiencia_profissional": experiencia
    }


def extract_experience_and_education_info(pages: List[Document], page_index: int = 1) -> Dict[str, str]:
    """
    Extrai experiência e formação a partir de uma página específica de um documento PDF.

    Args:
        pages (List[Document]): Lista de páginas carregadas do PDF.
        page_index (int): Índice da página a ser usada. Padrão é 1.

    Returns:
        Dict[str, str]: Um dicionário com as chaves 'experiencia' e 'formacao'.
    """
    if page_index >= len(pages):
        raise IndexError(f"O índice {page_index} excede o número de páginas disponíveis ({len(pages)}).")

    texto = pages[page_index].page_content
    texto_limpo = clean_text(texto)

    partes = texto_limpo.split("FORMAÇÃO")
    if len(partes) >= 2:
        # experiencia = partes[0].strip()
        formacao = partes[1].strip()
    else:
        # experiencia = texto_limpo
        formacao = ""

    return {
        # "experiencia": experiencia,
        "formacao": formacao
    }

def transform_dict(data: dict) -> dict:
    experience_part_one = data["informacoes_pessoais_experiencia"]["experiencia_profissional"]
    # experience_part_two = data["experiencia_formacao"]["experiencia"]
    academic_background = data["experiencia_formacao"]["formacao"]
    personal_information = data["informacoes_pessoais_experiencia"]["informacoes_pessoais"]

    return {
        "informacoes_pessoais": personal_information,
        "experiencia_profissional": experience_part_one,
        "formacao": academic_background
    }



def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("utf-8")


def start_faiss(api_key: str, data_dict: dict):
    persist_directory = 'data'
    embedding = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", "; ", " ", "\t", ""]
    )

    docs = []
    for key, text in data_dict.items():
        if not isinstance(text, str) or not text.strip():
            continue  # ignora campos vazios ou inválidos

        section_key = key.lower().replace(" ", "_")  # ex: 'informacoes_pessoais'
        document = Document(page_content=text.strip(), metadata={"secao": section_key})
        chunks = r_splitter.split_documents([document])
        docs.extend(chunks)
        print(f"[FAISS] Gerados {len(chunks)} chunks para seção '{section_key}'")

    print(f"[FAISS] Total de chunks processados: {len(docs)}")
    os.makedirs("data", exist_ok=True)
    vector_store = FAISS.from_documents(
        documents=docs, embedding=embedding
    )
    vector_store.save_local(persist_directory)
    db = FAISS.load_local(persist_directory, embeddings=embedding, allow_dangerous_deserialization=True)

    return db

def research_faiss(api_key: str, filter_to_db: str, question: str):
    docs = []
    embedding = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")
    db = FAISS.load_local("data", embeddings=embedding, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5, "filter": {"secao": filter_to_db}})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def api_chat(question: str, api_key: str, context: str):
    print(context)
    system = (f"""
    Você é um assistente que responde somente perguntas sobre o curriculum vitae. 
    Se a consulta não puder ser respondida com base no contexto fornecido ou se não souber a resposta, 
    basta dizer que você não sabe  a resposta, não tente inventar uma resposta. Mantenha a resposta o mais concisa 
    possível.
    """)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key, request_timeout=30)

    user = (f"contexto: {context} "
            f"pergunta: {question}")

    response = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])

    res = response.content
    return res

