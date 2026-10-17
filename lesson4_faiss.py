from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# 1. Read the document
with open("college_data.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 2. Split into sections
sections = text.split("\n\n")

chunks = []

for i in range(0, len(sections), 2):
    heading = sections[i]
    content = sections[i + 1]

    chunk = heading + "\n\n" + content
    chunks.append(chunk)


# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Convert chunks into embeddings
embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


# 5. Convert embeddings to NumPy float32
embeddings = np.array(embeddings).astype("float32")


# 6. Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


print("Number of vectors in FAISS:", index.ntotal)


# 7. Ask a question
query = "What percentage of attendance is required?"


# 8. Convert question into embedding
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")


# 9. Search FAISS
k = 2

distances, indices = index.search(query_embedding, k)


# 10. Display results
print("\nQuestion:")
print(query)

print("\nRetrieved chunks:")

for rank, (distance, index_number) in enumerate(
    zip(distances[0], indices[0]), start=1
):
    print(f"\n--- Result {rank} ---")
    print("Distance:", distance)
    print(chunks[index_number])