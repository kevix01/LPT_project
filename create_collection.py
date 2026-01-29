import os
import glob
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance


class CollectionCreator:
    """Crea e popola collezioni Qdrant con documenti PDF"""

    def __init__(self, embedding_model: str = "gemini"):
        """
        Inizializza il creatore di collezioni.

        Args:
            embedding_model: 'gemini' per GoogleGenerativeAI o 'hf' per HuggingFace
        """
        self.embedding_model_type = embedding_model

        # Setup embeddings
        self._setup_embeddings()

        # Setup Qdrant client
        self.client = QdrantClient(url="http://localhost")

    def _setup_embeddings(self):
        """Configura il modello di embedding"""
        if self.embedding_model_type == "gemini":
            os.environ["GOOGLE_API_KEY"] = "AIzaSyC24Wq37BNGGbO-qDOR1MR28UQ93BPhOd4"
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            print("Usando il modello embedding: gemini-embedding-001")
        elif self.embedding_model_type == "hf":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            print("Usando il modello embedding: all-MiniLM-L6-v2")
        else:
            raise ValueError("embedding_model deve essere 'gemini' o 'hf'")

    def load_pdfs_from_folder(self, folder_path: str = "./docs") -> List:
        """
        Carica tutti i PDF da una cartella.

        Args:
            folder_path: Percorso della cartella contenente i PDF

        Returns:
            Lista di documenti caricati
        """
        # Trova tutti i file PDF nella cartella
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"Nessun file PDF trovato in {folder_path}")

        print(f"Trovati {len(pdf_files)} file PDF:")
        for pdf in pdf_files:
            print(f"  - {os.path.basename(pdf)}")

        # Carica tutti i documenti
        all_docs = []
        for pdf_file in pdf_files:
            print(f"Caricamento: {os.path.basename(pdf_file)}...")
            # loader = PyPDFLoader(pdf_file)
            loader = PDFPlumberLoader(pdf_file)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  Caricate {len(docs)} pagine")

        return all_docs

    def split_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """
        Divide i documenti in chunk.

        Args:
            documents: Lista di documenti da dividere
            chunk_size: Dimensione di ogni chunk in caratteri
            chunk_overlap: Sovrapposizione tra chunk

        Returns:
            Lista di chunk
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

        all_splits = text_splitter.split_documents(documents)
        print(f"Documenti divisi in {len(all_splits)} chunk")

        return all_splits

    def create_collection(self, collection_name: str, documents: List):
        """
        Crea una nuova collezione Qdrant e la popola con i documenti.

        Args:
            collection_name: Nome della nuova collezione
            documents: Lista di documenti da indicizzare
        """
        # Verifica se la collezione esiste già
        if self.client.collection_exists(collection_name):
            overwrite = input(f"La collezione '{collection_name}' esiste già. Sovrascriverla? (s/n): ")
            if overwrite.lower() != 's':
                print("Operazione annullata.")
                return
            else:
                # Elimina la collezione esistente
                self.client.delete_collection(collection_name)
                print(f"Collezione '{collection_name}' esistente eliminata.")

        # Calcola la dimensione dei vettori
        sample_embedding = self.embeddings.embed_query("sample text")
        vector_size = len(sample_embedding)

        # Crea la collezione
        print(f"Creazione collezione '{collection_name}'...")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        # Crea il vector store
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )

        # Aggiungi i documenti
        print("Aggiunta documenti alla collezione...")
        document_ids = vector_store.add_documents(documents=documents)

        print(f"✓ Collezione '{collection_name}' creata con successo!")
        print(f"  - Documenti aggiunti: {len(document_ids)}")
        print(f"  - Dimensione vettori: {vector_size}")

        return vector_size, len(document_ids)

    def create_collection_from_pdfs(self, collection_name: str, folder_path: str = "./docs"):
        """
        Crea una collezione a partire da tutti i PDF in una cartella.

        Args:
            collection_name: Nome della nuova collezione
            folder_path: Percorso della cartella contenente i PDF
        """
        print(f"\n=== Creazione Collezione: {collection_name} ===")

        # 1. Carica PDF
        documents = self.load_pdfs_from_folder(folder_path)

        # 2. Divide in chunk
        splits = self.split_documents(documents)

        # 3. Crea collezione
        self.create_collection(collection_name, splits)


def main():
    """Funzione principale per la creazione di collezioni"""
    print("=== Creatore Collezioni Qdrant ===\n")

    # Scelta del modello di embedding
    print("Scegli il modello di embedding:")
    print("1. [gemini] Gemini (gemini-embedding-001) - richiede API key Google")
    print("2. [hf]     HuggingFace (all-MiniLM-L6-v2) - locale")

    embedding_model = input("\n").strip()

    # Inizializza il creatore
    creator = CollectionCreator(embedding_model=embedding_model)

    # Nome della collezione
    collection_name = input("\nInserisci il nome per la nuova collezione: ").strip()

    # Percorso della cartella
    folder_path = input("Percorso della cartella contenente i PDF [default: ./docs]: ").strip()
    if not folder_path:
        folder_path = "./docs"

    # Verifica che la cartella esista
    if not os.path.exists(folder_path):
        print(f"Errore: La cartella '{folder_path}' non esiste.")
        print("Assicurati che la cartella esista e contenga file PDF.")
        return

    try:
        # Crea la collezione
        creator.create_collection_from_pdfs(collection_name, folder_path)

        print("\n✓ Operazione completata con successo!")
        print(f"\nPer usare questa collezione con il sistema RAG:")
        print(f"  python rag_system.py")
        print(f"  > Inserisci il nome della collezione: {collection_name}")

    except Exception as e:
        print(f"Errore durante la creazione della collezione: {e}")
        print("\nAssicurati che:")
        print("1. Qdrant sia in esecuzione (docker run -p 6333:6333 qdrant/qdrant)")
        print("2. La cartella contenga file PDF validi")
        print("3. Per Gemini: l'API key sia configurata correttamente")


if __name__ == "__main__":
    main()