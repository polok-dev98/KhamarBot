# index_builder.py

import os
import faiss
import pickle
import json
import glob
import numpy as np
from tqdm import tqdm
from openai import AzureOpenAI
from dotenv import load_dotenv

class KnowledgeEmbedder:
    def __init__(self, json_dir="./pdf_file"):
        self.json_dir = json_dir
        self.embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_EMBEDDING_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_EMBEDDING_URL")
        )
        self.embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.index = None
        self.knowledge_data = []

    def _get_embeddings(self, texts):
        """Get embeddings for a batch of texts"""
        embeddings = []
        batch_size = 16
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i+batch_size]
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_deployment,
                    input=batch
                )
                batch_embeddings = [np.array(data.embedding).astype('float32') for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Add zero vectors for failed batches
                embeddings.extend([np.zeros(1536).astype('float32')] * len(batch))
        
        return np.array(embeddings)

    def _split_text(self, text, chunk_size=800, chunk_overlap=100):
        """Simple text splitter"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find chunk end
            end = start + chunk_size
            
            # If we're at the end
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to split at sentence boundary
            split_point = text.rfind('. ', start, end)
            if split_point == -1:
                split_point = text.rfind('। ', start, end)  # Bengali full stop
            if split_point == -1:
                split_point = text.rfind(' ', start, end)
            if split_point == -1 or split_point < start + chunk_size * 0.7:
                split_point = end
            
            chunks.append(text[start:split_point].strip())
            start = split_point - chunk_overlap
        
        return chunks

    def process_json(self):
        """Load and process JSON files"""
        print(f"🔍 Loading JSON files from: {self.json_dir}")
        
        json_files = glob.glob(os.path.join(self.json_dir, "*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {self.json_dir}")
            return
        
        print(f"📂 Found {len(json_files)} JSON files")
        
        all_texts = []
        all_metadata = []
        
        for json_file in tqdm(json_files, desc="Processing files"):
            filename = os.path.basename(json_file)
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Process each entry
                for idx, entry in enumerate(data):
                    content = entry.get('content', '')
                    header = entry.get('header', '')
                    page = entry.get('page', '')
                    
                    # Create text with metadata
                    text_with_context = f"File: {filename}"
                    if header:
                        text_with_context += f"\nHeader: {header}"
                    if page:
                        text_with_context += f"\nPage: {page}"
                    text_with_context += f"\nContent: {content}"
                    
                    # Split into chunks
                    chunks = self._split_text(text_with_context)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        # Create combined text for embedding (like CSV's combined_text)
                        combined_text = f"{header or filename}. {chunk}"
                        all_texts.append(combined_text)
                        
                        # Store metadata (matching your CSV structure)
                        metadata = {
                            "ki_topic": header or filename,
                            "ki_text": chunk,
                            "source_file": filename,
                            "page": str(page),
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks)
                        }
                        all_metadata.append(metadata)
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"📊 Total documents (chunks): {len(all_texts)}")
        
        if not all_texts:
            print("❌ No text extracted from JSON files")
            return
        
        # Generate embeddings
        print(f"🔢 Generating embeddings for {len(all_texts)} entries...")
        embeddings = self._get_embeddings(all_texts)
        
        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        
        # Save metadata for retrieval
        self.knowledge_data = all_metadata
        print("✅ Embedding complete.")

    def save_to_disk(self, index_path="vector_store/kb_index.faiss", metadata_path="vector_store/kb_metadata.pkl"):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        print(f"💾 Saved FAISS index to: {index_path}")

        with open(metadata_path, "wb") as f:
            pickle.dump(self.knowledge_data, f)
        print(f"💾 Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    load_dotenv()
    
    # Use JSON directory instead of CSV
    embedder = KnowledgeEmbedder(json_dir="./pdf_file")
    embedder.process_json()
    embedder.save_to_disk()