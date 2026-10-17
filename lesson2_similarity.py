from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Students need at least 75 percent attendance.",
    "The hostel fee is 80000 rupees per year.",
    "Semester examinations are conducted twice a year.",
    "Students can apply for placements after meeting eligibility requirements."
]

query = "What percentage of attendance do I need?"

document_embeddings = model.encode(documents)
query_embedding = model.encode([query])

similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

for document, score in zip(documents, similarities):
    print(f"{score:.4f} -> {document}")