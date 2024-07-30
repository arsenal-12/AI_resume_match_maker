from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Load the text file
    loader = TextLoader("resume.txt")
    documents = loader.load()
    print(type(documents))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    # Load the embedding model 
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
print(embeddings)
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="testing_db"
)

print("Vector DB Successfully Created!")