import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY");

openai_ef = OpenAIEmbeddingFunction(api_key=openai_key, model_name="text-embedding-3-small")

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

client = OpenAI(api_key=openai_key)

# res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "What is the capital of Vietnam?"}])

import os

def load_documents_from_directory(path):
    documents = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding='utf-8') as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

def split_text(text, size=1000, overlapSize=20) :
    texts = []
    start = 0
    while start < len(text):
        end = start + size
        texts.append(text[start:end])
        start = end - overlapSize
    return texts

def get_openai_embedding(text): 
    res = client.embeddings.create(input=text, model="text-embedding-3-small")
    return res.data[0].embedding

def query_documents(question, n_results = 2):
    # question_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results= n_results)

    # Extract the relevant chunks
    relevant_chunks = [chunk for sublist in results['documents'] for chunk in sublist]
    return relevant_chunks

def generate_response(question):
    prompt = (
        "You are an assistant for question-answering tasks. From what you was trained, "
        "use three sentences maximum and keep the answer concise. "
        "If you don't know the answer, say that you don't know."
    )
    messages = [{"role": "system","content": prompt},
                {"role": "user","content": question}]

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    answer = response.choices[0].message
    return answer

def generate_response_rag(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    messages = [{"role": "system","content": prompt},
                {"role": "user","content": question}]

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    answer = response.choices[0].message
    return answer


# Step 1: Loading documents from directory
documents = load_documents_from_directory("./articles");
print(f"{len(documents)} documents loaded")
print(f"-------------------------------------------------------")

# Step 2: Splitting documents and loading into chunks
chunks = []
for doc in documents:
    print(f"Loading into chunks for {doc['id']}")
    texts = split_text(doc["text"])
    for i, text in enumerate(texts):
        chunks.append({"id": f"{doc['id']}_chunk{i+1}", "text": text})
print(f"{len(chunks)} chunks have been loaded")
print(f"-------------------------------------------------------")

# Step 3: Generate embedding for each chunk
for chunk in chunks:
    chunk['embedding'] = get_openai_embedding(doc['text'])
    # print(chunk['embedding'])
print(f"{len(chunks)} embeddings have been generated")
print(f"-------------------------------------------------------")

# Step 4: Insert chunks along with embeddings
for chunk in chunks:
    collection.upsert(ids=[chunk['id']], documents=[chunk['text']], embeddings=chunk['embedding'])

# Step 5: Execute
question = "What is the special event occurs in Vietnam in 2024?"
answerFromLLM = generate_response(question)
answerFromRAG = generate_response_rag(question=question, relevant_chunks=query_documents(question))

print(f"Answer from LLM:")
print(answerFromLLM)
print(f"Answer from RAG:")
print(answerFromRAG)


