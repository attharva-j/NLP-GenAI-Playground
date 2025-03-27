import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import yaml
import os
import tiktoken

# Load credentials
with open("/Users/atjoshi/Desktop/CREDS/keys.yaml", "r") as file:
    configs = yaml.safe_load(file)

# Pinecone init
pc = Pinecone(api_key=configs['pinecone'])
index = pc.Index(configs['pinecone_index'])
# GPT-4 tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")


# Embeddings and LLM
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=2048,
    openai_api_key=configs["open_ai"]
)
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=configs["open_ai"])

# Helper: embed query
def get_embedding(text):
    return embedding_model.embed_query(text)

# Helper: retrieve top-k from Pinecone
def retrieve_context(query, k=8):
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return [
        {
            "content": match["metadata"].get("content", ""),
            "source": match["metadata"].get("file_name", "unknown")
        }
        for match in results["matches"]
    ]


def count_tokens(text):
    return len(encoding.encode(text))

def truncate_texts_by_token_limit(texts, max_tokens):
    total_tokens = 0
    result = []
    for text in texts:
        tokens = count_tokens(text)
        if total_tokens + tokens > max_tokens:
            break
        result.append(text)
        total_tokens += tokens
    return result


# Helper: build system prompt
def build_context_prompt(user_query, retrieved_chunks):
    # context = "\n---\n".join(chunk.get("text", "") or chunk.get("content", "") for chunk in retrieved_chunks)
    raw_contexts = [chunk.get("content", "") for chunk in retrieved_chunks]
    trimmed_contexts = truncate_texts_by_token_limit(raw_contexts, max_tokens=8000)
    context = "\n---\n".join(trimmed_contexts)

    # history = "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in chat_history])
    return f"""You are a financial analyst assistant. Use the following context to help answer the user's question.

    Context:
    {context}
    
    User Question:
    {user_query}
    
    If the context contains relevant information, use it to answer the user's question. If the answer isn't directly present, try to make a helpful and reasonable inference based on the data. Do not decline to answer unless you truly cannot help.
    
    
    Now, answer the user's latest question clearly and accurately."""

# Streamlit UI
st.set_page_config(page_title="FinRAG Chatbot", layout="wide")
st.title("💬 FinRAG Chatbot")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_input = st.chat_input("Ask a financial question based on uploaded files...")

if user_input:
    # # Gather context from last 5 interactions
    # history_context = st.session_state.history[-5:]
    #
    # # Combine last user messages and current one
    # all_texts = [msg["user"] for msg in history_context] + [user_input]
    combined_query = " ".join(user_input)


    st.write(f"Query given for context retrieval: {combined_query}")
    # Retrieve documents from Pinecone
    retrieved_docs = retrieve_context(combined_query)
    # st.write("🔍 Retrieved Context:", retrieved_docs)

    # Build prompt
    prompt = build_context_prompt(combined_query, retrieved_docs)

    messages = [
        {"role": "system", "content": "You are a helpful financial assistant who knows everything about financial state "
                                      "of SAIA Inc."},
        {"role": "user", "content": build_context_prompt(user_input, retrieved_docs)}
    ]

    response = llm.invoke(messages)

    # Get LLM response
    response = llm.invoke(prompt)

    # Save to history
    st.session_state.history.append({"user": user_input, "bot": response.content})

# Show chat messages
for msg in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])
