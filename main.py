from typing import Union, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from embedder import ask, generate_index
app = FastAPI()


TOKEN_VALIDATION_ERROR = "Validation error: Provided token error."
INDEX_UPDATED_SUCCESSFULLY = "Index was updated successfully."


class PostgresConnection(BaseModel):
    scheme: str = Field(default="postgres", description="Scheme for the connection, default is 'postgres'")
    dbname: str = Field(..., description="Name of the database")
    user: str = Field(..., description="Username for the database")
    password: str = Field(..., description="Password for the database")
    host: str = Field(..., description="Host where the database is located")
    port: int = Field(..., description="Port number for the database connection")
    tablename: str = Field(..., description="Name of the table to interact with")

class PromptRequest(BaseModel):
    query: Union[str, None] = Field(..., description="Query string")
    token: Union[str, None] = Field(..., description="Authentication token")
    postgres_connection: PostgresConnection = Field(..., description="Postgres connection details")


def validate_token(token):
    # TODO
    return True


app = FastAPI()


@app.post("/ask/")
async def create_item(prompt: PromptRequest):
    if validate_token(prompt.token):
        return ask(prompt.query, prompt.postgres_connection)
    else:
        return TOKEN_VALIDATION_ERROR


@app.post("/update-index/")
async def create_item(prompt: PromptRequest):
    if validate_token(prompt.token):
        generate_index(prompt.postgres_connection)
        return INDEX_UPDATED_SUCCESSFULLY
    else:
        return TOKEN_VALIDATION_ERROR
