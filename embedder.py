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


load_dotenv()


QUERY = f"""
SELECT e.campo, CONCAT('CODIGO', e.id) FROM tabla e;
"""


def fetch_documents_from_storage(query: str, connection):
    # Prep documents - fetch from DB
    DatabaseReader = download_loader("DatabaseReader")
    reader = DatabaseReader(
        scheme=connection.scheme,  # Database Scheme
        dbname=connection.dbname,  # Database Name
        user=connection.user,  # Database User
        password=connection.password,  # Database Password
        host=connection.host,  # Database Host
        port=connection.port,  # Database Port
    )
    return reader.load_data(query=query)



# You would cache this
EMBED_MODEL = HuggingFaceEmbedding(
    model_name="WhereIsAI/UAE-Large-V1",
    embed_batch_size=10,  # open-source embedding model
)


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
        documents=fetch_documents_from_storage(QUERY, connection),
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
