from excelparser import ExcelAdapter

import yaml

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from typing import List, Dict, Any, Optional
from pinecone import Pinecone
import os
from tqdm import tqdm
# from langchain.chains.question_answering import load_qa_chain
# import asyncio
import uuid
import warnings
warnings.filterwarnings("ignore")

with open("/Users/atjoshi/Desktop/CREDS/keys.yaml", "r") as file:
    configs = yaml.safe_load(file)


def connect_pinecone_and_get_index():
    pc = Pinecone(api_key=configs['pinecone'])
    index = pc.Index(configs['pinecone_index'])

    return pc, index


def get_raw_files_from_source(path: str) -> list[str]:
    return [file for file in os.listdir(path) if file not in ['.DS_Store']]


def parse_raw_files(path_to_raw_file_directory: str) -> list[dict]:
    files = get_raw_files_from_source(path_to_raw_file_directory)
    knowledge_base = []
    for file in files:
        ext = file.split('.')[-1]
        print(f"Current file format: {ext}")
        if ext in ['xls', 'xlsx']:
            with open(f"{path_to_raw_file_directory}{file}", "rb") as f:
                file_bytes = f.read()
            result = ExcelAdapter().parse(file_bytes, {'file_name': file})
            knowledge_base.extend(result)
            continue
        else:
            print(f"File format {ext} is not currently supported for parsing.")
            continue
    return knowledge_base


def smart_chunk_documents(documents: List[Dict[str, str]], max_lines_per_chunk: int = 10) -> List[Dict[str, str]]:
    chunked_documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        if metadata.get("file_type") == "excel_sheet":
            lines = content.splitlines()

            if len(lines) <= max_lines_per_chunk:
                # Keep as-is if short enough
                chunked_documents.append({
                    "content": content,
                    "metadata": metadata
                })
            else:
                # Break into N-line chunks
                for i in range(0, len(lines), max_lines_per_chunk):
                    chunk_lines = lines[i:i + max_lines_per_chunk]
                    chunk_text = "\n".join(chunk_lines)
                    chunked_documents.append({
                        "content": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": i // max_lines_per_chunk
                        }
                    })
        else:
            # Use recursive character-based chunking for other file types
            splits = text_splitter.split_text(content)
            for i, chunk in enumerate(splits):
                chunked_documents.append({
                    "content": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": i
                    }
                })

    return chunked_documents


def embed_texts_with_openai(texts: list[str]) -> list[list[float]]:
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=2048,
        openai_api_key=configs["open_ai"]
    )
    return embedding_model.embed_documents(texts)

def upsert_chunks_to_pinecone(index, chunks: list[dict], batch_size=100):
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [chunk["content"] for chunk in batch]
        embeddings = embed_texts_with_openai(texts)

        # Prepare for upsert
        vectors = []
        for chunk_data, vector in zip(batch, embeddings):
            vector_id = str(uuid.uuid4())
            metadata = chunk_data["metadata"]
            vectors.append({
                "id": vector_id,
                "values": vector,
                "metadata": {
                    **metadata,
                    "content": chunk_data["content"]
                }
            })

        index.upsert(vectors=vectors)
    print("Knowledge base embedded and upserted in Pinecone vector store.")

if __name__ == '__main__':
    kb = parse_raw_files("./RAWDATA/")
    chunked_docs = smart_chunk_documents(kb)
    pc, index = connect_pinecone_and_get_index()
    upsert_chunks_to_pinecone(index, chunks=chunked_docs)

