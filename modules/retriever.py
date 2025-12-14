# retriever.py

import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity

class VectorRetriever:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Don't initialize client immediately
        self._embedding_client = None
        self._embedding_deployment = None
        
        self.index = None
        self.knowledge_data = None
        self._load_index_and_metadata()

    def _get_embedding_client(self):
        """Get Azure OpenAI client"""
        if self._embedding_client is None:
            api_key = os.getenv("AZURE_EMBEDDING_KEY")
            azure_endpoint = os.getenv("AZURE_EMBEDDING_URL")
            
            if not api_key or not azure_endpoint:
                raise ValueError("Missing Azure OpenAI credentials")
            
            self._embedding_client = AzureOpenAI(
                api_key=api_key,
                api_version="2023-05-15",
                azure_endpoint=azure_endpoint
            )
            self._embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        
        return self._embedding_client, self._embedding_deployment

    def _load_index_and_metadata(self):
        """Load FAISS index and metadata"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
        with open(self.metadata_path, "rb") as f:
            self.knowledge_data = pickle.load(f)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Azure OpenAI"""
        client, deployment = self._get_embedding_client()
        
        response = client.embeddings.create(
            model=deployment,
            input=text
        )
        return np.array(response.data[0].embedding).astype('float32')

    def retrieve(self, query: str, top_k: int = 3, similarity_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector store"""
        query_embedding = self._get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.knowledge_data):
                # Compute cosine similarity
                doc_embedding = self.index.reconstruct(int(idx)).reshape(1, -1)
                sim = cosine_similarity(query_embedding, doc_embedding)[0][0]
                if sim >= similarity_threshold:
                    results.append({
                        **self.knowledge_data[idx],
                        "similarity": float(sim)
                    })

        return results

    def as_tool(self, query: str) -> str:
        """Tool function for the agent to call"""
        print(f"\n🔍 [RETRIEVER DEBUG] Query: '{query}'")
        
        docs = self.retrieve(query, top_k=5, similarity_threshold=0.4)
        
        if not docs:
            print(f"❌ [RETRIEVER DEBUG] No relevant documents found")
            return "No relevant documents found in knowledge base."
        
        # Show retrieved docs in terminal with book name and page number
        print(f"✅ [RETRIEVER DEBUG] Found {len(docs)} relevant documents:")
        for i, doc in enumerate(docs):
            print(f"   📄 Document {i+1} (Similarity: {doc['similarity']:.3f})")
            print(f"      Topic: {doc['ki_topic']}")
            
            # Show book name (source_file) if available
            book_name = doc.get('source_file', 'Unknown Book')
            print(f"      Book: {book_name}")
            
            # Show page number if available
            if doc.get('page') and doc['page'] != '' and doc['page'] != 'None':
                print(f"      Page: {doc['page']}")
            
            # Show chunk info if available
            if doc.get('chunk_index') is not None and doc.get('total_chunks'):
                print(f"      Chunk: {doc['chunk_index'] + 1}/{doc['total_chunks']}")
            
            # Show content preview
            content_preview = doc['ki_text'][:150] + "..." if len(doc['ki_text']) > 150 else doc['ki_text']
            print(f"      Content: {content_preview}")
            print()
        
        # Create context for the agent (include all metadata)
        context_parts = []
        for i, doc in enumerate(docs):
            context = f"Document {i+1} (Similarity: {doc['similarity']:.2f}):\n"
            context += f"Topic: {doc['ki_topic']}\n"
            
            # Add book name to context for the agent
            if doc.get('source_file'):
                context += f"Source: {doc['source_file']}\n"
            
            # Add page number to context for the agent
            if doc.get('page') and doc['page'] != '' and doc['page'] != 'None':
                context += f"Page: {doc['page']}\n"
            
            # Add chunk info
            if doc.get('chunk_index') is not None and doc.get('total_chunks'):
                context += f"Part: {doc['chunk_index'] + 1} of {doc['total_chunks']}\n"
            
            context += f"Content: {doc['ki_text']}"
            context_parts.append(context)
        
        context = "\n\n".join(context_parts)
        
        return context