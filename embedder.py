from llama_index import (
    download_loader
)
from llama_index.embeddings import HuggingFaceEmbedding
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
from llama_index.vector_stores import PGVectorStore

from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.llms.ollama import Ollama

from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    Document,
)
from llama_index.vector_stores import PGVectorStore


from main import PostgresConnection

load_dotenv()


query = f"""
SELECT e.campo, CONCAT('CODIGO', e.id) FROM tabla e;
"""


def fetch_documents_from_storage(query: str):
    # Prep documents - fetch from DB
    DatabaseReader = download_loader("DatabaseReader")
    reader = DatabaseReader(
        scheme=os.environ.get("db_scheme"),  # Database Scheme
        dbname=os.environ.get("db_name"),  # Database Name
        user=os.environ.get("db_user"),  # Database User
        password=os.environ.get("db_password"),  # Database Password
        host=os.environ.get("db_host"),  # Database Host
        port=os.environ.get("db_port"),  # Database Port
    )
    return reader.load_data(query=query)


documents = fetch_documents_from_storage(query=query)

# You would cache this
EMBED_MODEL = HuggingFaceEmbedding(
    model_name="WhereIsAI/UAE-Large-V1",
    embed_batch_size=10,  # open-source embedding model
)

# query = """
# UPDATE tabla
# SET embedding = '{}'
# WHERE id = '{}';
# """


# def save_embeddings(documents, query) -> None:
#     for document in documents:
#         text = document.text.split("$$$", 1)[0][:-2]
#         uuid = document.text.split("$$$", 1)[1]
#         # print(f"{uuid}: {text}")
#         embedding = EMBED_MODEL.get_text_embedding(text)
#         query = query.format(embedding, uuid)
#         # Conectar con la db y ejecutar la query
#         conn = psycopg2.connect(
#             dbname=os.environ.get("db_name"),
#             user=os.environ.get("db_user"),
#             password=os.environ.get("db_password"),
#             host=os.environ.get("db_host"),
#             port=os.environ.get("db_port"),
#         )
#         cursor = conn.cursor()
#         cursor.execute(query)
#         conn.commit()


# save_embeddings(documents, query)  # documents loaded previously

llm = Ollama(
    base_url="localhost:11434",  # Download the GGUF from hugging face
    model="llama3.1",
)
embed_model = HuggingFaceEmbedding(
    model_name="WhereIsAI/UAE-Large-V1",
    embed_batch_size=10,  # open-source embedding model
)
hf_token = "hf_"

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    system_prompt="Sos una insteligencia artificial que contesta en español.",
)


def _generate_vector_store_index(
    vector_store: PGVectorStore,
    documents,
    service_context: ServiceContext,
) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )


def _vector_store(connection):
    vector_store = PGVectorStore.from_params(
        database=connection.dbname,
        user=connection.user,
        password=connection.password,
        host=connection.host,
        port=connection.port,
        table_name=connection.tablename,
        embed_dim=int(os.environ.get("embedding_dimensions")),
    )
    return vector_store


def generate_index(connection):
    _generate_vector_store_index(
        documents=documents,
        service_context=service_context,
        vector_store=_vector_store(connection),
    )


def ask(prompt, connection):
    index = VectorStoreIndex.from_vector_store(
        vector_store=_vector_store(connection), service_context=service_context
    )
    query_engine = index.as_query_engine(
        similarity_top_k=3,
    )
    response = query_engine.query(prompt)
    return response
