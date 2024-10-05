import os
import umap
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from helper_utils import (load_documents_from_directory, project_embeddings)
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import (RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter,)

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY");
client = OpenAI(api_key=openai_key)

# Step 1: Loading documents from directory
documents = load_documents_from_directory("./articles");
print(f"{len(documents)} documents loaded")
print(f"-------------------------------------------------------")

# Step 2: Map documents to texts
texts = [];
for doc in documents: 
    texts += [doc['text']]
print(f"{len(texts)} texts mapped")
print(f"-------------------------------------------------------")

# Step 3: Concat texts together and split by character to get sentences
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(texts))
print(len(character_split_texts))

# Step 4: Loop thru all texts and continue to split them by token (chunk size)
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=100
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Step 5: Insert into vector database along with embeddings
ids = [str(i) for i in range(len(token_split_texts))]
embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    name="query-expansion-collection", embedding_function=embedding_function
)
chroma_collection.add(ids=ids, documents=token_split_texts)

# Step 6: Create original query and augmented query
original_query = "What was the typhoon which just occured. How many countries it effected?"

prompt = "You are a helpful assistant. Provide an example answer to the given question, that might be found in a document."
messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": original_query},
    ]
response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
hallucinatedAnswer = response.choices[0].message.content
print(hallucinatedAnswer)
augmented_query = f"{original_query} {hallucinatedAnswer}"

# Step 7: Compare the quality between original query and augmented query (query-expansion)
# embeddings in vector database (all)
documents = chroma_collection.get(include=["embeddings"])
embeddings = documents["embeddings"]

# embedding of original query
original_query_embedding = embedding_function([original_query])
# embedding of augmented query
augmented_query_embedding = embedding_function([augmented_query])

# embeddings retrieved when query with augmented query
retrieved_documents = chroma_collection.query(query_texts=augmented_query, n_results=5, include=["documents", "embeddings"])
retrieved_embeddings = retrieved_documents["embeddings"][0]


umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)

# datasets embeddings in vector database (all)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
# datasets embedding of original query
projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
# datasets embedding of augmented query
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
# datasets embeddings retrieved when query with augmented query
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

import matplotlib.pyplot as plt
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot
