import chromadb
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, PlaywrightURLLoader
from langchain_text_splitters.character  import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from typing import List


class rag_Pipeline:
    """
    RAG Pipeline
    """

    def  __init__(self):
        self.rag = chromadb.Client()
        self.embedding = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            # model_kwargs={"attn_implementation": "flash_attention_2"},  # could be faster
            tokenizer_kwargs={"padding_side": "left"}
        )
        self.collection = self.rag.get_or_create_collection(name="default")
        self.library = None
        self.chunks = None
        
        self.loadData = self.save_embeddings()
        
    def create_library(self):
        
        path = Path(__file__)
        pdf_files = list(path.glob('*.pdf'))
        docs = [PyMuPDFLoader.load(file) for file in pdf_files]
        
        
        urls = ['https://support.geotab.com/mygeotab/doc/safety-center']
        loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer", "nav"])
        documents = loader.load()
        # documents_content = [document.page_content for document in documents]
        
        self.library = docs + documents 
        
    def split_into_chunks(self):
        
        self.create_library()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
        chunks = text_splitter.split_documents(self.library)
        self.chunks = [chunk.page_content for chunk in chunks]
        
    def create_embeddings(self, query: str = None) -> List[List[float]]:
        
        # embedding = self.embedding_model.encode(chunks,normalize_embeddings=True,prompt_name="query")
        # return embedding.tolist()
        
        if query:
            
            return [self.embedding.encode(query,normalize_embeddings=True,prompt_name="query").tolist()]

        # SentenceTransformer will need string to embedding encode
        return [self.embedding.encode(chunk,normalize_embeddings=True,prompt_name="query").tolist() for chunk in self.chunks]
    
    
    def save_embeddings(self) -> None:
        """
        save to ChromaDB 
        """
        
        self.split_into_chunks()
        
        
        ids = []
        documents = []
        metas = []
        embs = []
        
        for i, (chunk, embedding) in enumerate(zip(self.chunks, self.create_embeddings())):
            ids.append(str(i))                    # if this ID is not unique, ChromaDB will throw an error
            documents.append(chunk)
            metas.append({"index": str(i)})       # meaningful metadata
            embs.append(embedding)
        
        # add the chromadb in a batch
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metas,
            embeddings=embs,
        )
        
        
    def retrieve(self,query:str, top_k:int) -> List[str]:
        query_embedding = self.create_embeddings(query)
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
            )
        # chromadb already have similarity search，ranking from high score to low score
        # print(results['documents'])
        return results['documents'][0] # since this is a nested list: [[123, 456]]


    def rerank(self,query: str, retrieved_chunks:List[str], top_k:int) -> List[str]:
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') # for english reranker
        
        pairs = [(query,chunk) for chunk in retrieved_chunks]
        scores = cross_encoder.predict(pairs)

        scored_chunks = list(zip(retrieved_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1],reverse=True)

        return [chunk for chunk, _ in scored_chunks][:top_k] #返回排行前top_k高的retrieve回答

    # reranked_chunks = rerank(query,retrieved_chunks,3)
    # print(reranked_chunks)
        

if __name__ == "__main__":        
    rag_test = rag_Pipeline()


    retrieved_chunks = rag_test.retrieve("what is safety?",5)
    reranked_chunks = rag_test.rerank("what is safety?",retrieved_chunks,3)
    print(reranked_chunks)
        
        
        
    