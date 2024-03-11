from haystack import Pipeline
from haystack.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import Pipeline
from haystack.schema import Document
from pymongo import MongoClient
import torch

mongodb_conn_string = 'mongodb+srv://<username>:<password>@maincluster.d67gxdl.mongodb.net/'
db_name = "<db_name>"
collection_name = "<collection_name>"
index_name = "<vsearch_index_name>"
embedding_model_path=r"C:\Code\Python\Models\all-mpnet-base-v2"

# Initialize MongoDB python client
client = MongoClient(mongodb_conn_string)
collection = client[db_name][collection_name]

# Reset w/out deleting the Search Index 
collection.delete_many({})

JsonKnowledgeObject = {}
# Adding Content to be indexed
JsonKnowledgeObject[
    "content"
] = """Faiss (Facebook AI Similarity Search) is an open-source library developed by Facebook, designed for efficient similarity searches and clustering of dense vectors. This library addresses challenges commonly encountered in machine learning applications, particularly those involving high-dimensional vectors, such as image recognition and recommendation systems. Its widespread applicability, combined with features like scalability and flexibility, makes it a valuable tool for various machine learning and data analysis tasks, as demonstrated in its real-world application scenarios outlined in the Facebook Engineering blog post. 

Faiss employs advanced techniques like indexing and quantization to accelerate similarity searches in large datasets. Its versatility is evident in its support for both CPU and GPU implementations, ensuring scalability across different hardware configurations. Faiss offers flexibility with options for both exact and approximate similarity searches, allowing users to tailor the level of precision to their specific requirements."""

# Adding Meta Data
JsonKnowledgeObject["meta"] = {}
JsonKnowledgeObject["meta"][
    "title"
] = "Semantic Search With Facebook AI Similarity Search (FAISS)"
JsonKnowledgeObject["meta"]["author"] = "ThreadWaiting"
JsonKnowledgeObject["meta"][
    "link"
] = "https://threadwaiting.com/semantic-search-with-facebook-ai-similarity-search-faiss/"


# Convert Json object to Document object
document = Document(
    content=JsonKnowledgeObject["content"], meta=JsonKnowledgeObject["meta"]
)
# use GPU if available and drivers are installed
use_gpu = True if torch.cuda.is_available() else False

document_store = MongoDBAtlasDocumentStore(
    mongo_connection_string=mongodb_conn_string,
    database_name=db_name,
    collection_name=collection_name,
    vector_search_index=index_name,
    similarity="cosine",
    embedding_dim=768,
    embedding_field = "embedding"
)

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model=embedding_model_path,
    model_format="sentence_transformers",
    top_k=10,
)



# Add document to the document store
document_store.write_documents([document])
# This needs to be executed every time the data gets refreshed
retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model=embedding_model_path, use_gpu=use_gpu
)
document_store.update_embeddings(retriever)

retriever = retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model=embedding_model_path, use_gpu=use_gpu,
    model_format="sentence_transformers",
    top_k=3
)

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])

output = query_pipeline.run(query="What is Faiss?")

results_documents = output["documents"]

if len(results_documents) > 0:
    print("\nMatching Article: \n")
    for doc in results_documents:
        docDoc = doc.to_dict()
        print(docDoc["meta"]["title"])
        print(docDoc["content"])
        score = round(float(str(docDoc["score"] or "0.0")) * 100, 2)
        print("Match score:", score, "%")