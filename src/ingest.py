import os
import re
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Any

# Imports LangChain avec loaders PDF spécialisés
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================
# CONFIGURATION
# ==========================
class Config:
    CHUNK_SIZE = 512  # Taille réduite pour les livres techniques
    CHUNK_OVERLAP = 50
    MIN_DOC_LENGTH = 10
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================
# SETUP LOGGING
# ==========================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

# ==========================
# PDF TEXT CLEANER
# ==========================
class PDFTextCleaner:
    """Nettoyeur spécialisé pour le texte extrait de PDF"""
    
    @staticmethod
    def clean_pdf_text(text: str) -> str:
        """Nettoie le texte des PDF selon les bonnes pratiques LangChain"""
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Suppression des caractères de contrôle et Unicode problématiques
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # 2. Nettoyage des caractères Unicode spéciaux fréquents dans les PDF
        text = re.sub(r'[\u200b-\u200f\u2028-\u202e\ufeff]', '', text)
        
        # 3. Gestion des césures de mots en fin de ligne
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # 4. Normalisation des espaces et retours à la ligne
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # 5. Suppression des en-têtes/pieds de page typiques des livres
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Ignorer les lignes trop courtes ou numéros de page
            if (len(line) > 3 and 
                not line.isdigit() and 
                not re.match(r'^[ivxlc]+$', line.lower())):
                cleaned_lines.append(line)
        
        text = ' '.join(cleaned_lines)
        
        return text.strip()

# ==========================
# PDF LOADER STRATEGY
# ==========================
class PDFLoaderStrategy:
    """Stratégie de chargement PDF avec fallbacks"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.cleaner = PDFTextCleaner()
    
    def load_pdf(self, file_path: Path) -> Optional[List[Any]]:
        """Charge un PDF avec différentes stratégies"""
        strategies = [
            self._try_pymupdf,  # Meilleur pour l'extraction texte
            self._try_pypdf,     # Fallback standard
        ]
        
        for strategy in strategies:
            try:
                documents = strategy(file_path)
                if documents and len(documents) > 0:
                    return documents
            except Exception as e:
                self.logger.warning(f"❌ {strategy.__name__} échoué: {e}")
                continue
        
        return None
    
    def _try_pymupdf(self, file_path: Path) -> Optional[List[Any]]:
        """Essaie PyMuPDF (meilleur pour la précision)"""
        try:
            # PyMuPDF est généralement meilleur pour l'extraction texte
            loader = PyMuPDFLoader(str(file_path))
            docs = loader.load()
            
            # Nettoyage approfondi
            for doc in docs:
                doc.page_content = self.cleaner.clean_pdf_text(doc.page_content)
            
            self.logger.info(f"✅ PyMuPDF: {len(docs)} pages")
            return docs
        except ImportError:
            self.logger.warning("📚 PyMuPDF non installé, utilisation de PyPDF")
            return None
    
    def _try_pypdf(self, file_path: Path) -> Optional[List[Any]]:
        """Essaie PyPDF (fallback)"""
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        
        # Nettoyage approfondi
        for doc in docs:
            doc.page_content = self.cleaner.clean_pdf_text(doc.page_content)
        
        self.logger.info(f"✅ PyPDF: {len(docs)} pages")
        return docs

# ==========================
# DOCUMENT PROCESSOR
# ==========================
class DocumentProcessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def create_chunks(self, documents: List[Any]) -> List[Any]:
        """Crée des chunks validés"""
        if not documents:
            return []
        
        # Validation des documents
        valid_docs = []
        for doc in documents:
            if (hasattr(doc, 'page_content') and 
                doc.page_content and 
                isinstance(doc.page_content, str)):
                
                # Validation de longueur
                if len(doc.page_content.strip()) >= Config.MIN_DOC_LENGTH:
                    valid_docs.append(doc)
        
        self.logger.info(f"📄 Documents valides: {len(valid_docs)}")
        
        if not valid_docs:
            return []
        
        # Création des chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        chunks = splitter.split_documents(valid_docs)
        
        # Validation finale
        valid_chunks = []
        for chunk in chunks:
            if (hasattr(chunk, 'page_content') and 
                chunk.page_content and 
                isinstance(chunk.page_content, str) and
                len(chunk.page_content.strip()) >= 10):
                
                valid_chunks.append(chunk)
        
        self.logger.info(f"✂️  Chunks valides: {len(valid_chunks)}")
        return valid_chunks

# ==========================
# VECTOR STORE BUILDER - VERSION SIMPLIFIÉE
# ==========================
class VectorStoreBuilder:
    def __init__(self, chroma_path: str, logger: logging.Logger):
        self.chroma_path = Path(chroma_path)
        self.logger = logger
        
        # Embedding model simple et stable
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
    
    def safe_build_vector_store(self, chunks: List[Any]) -> Optional[Chroma]:
        """Construction sécurisée de la base vectorielle"""
        if not chunks:
            self.logger.error("❌ Aucun chunk à vectoriser")
            return None
        
        # Nettoyage
        if self.chroma_path.exists():
            shutil.rmtree(self.chroma_path)
        
        try:
            self.logger.info("🔨 Construction base vectorielle...")
            
            # Méthode DIRECTE de LangChain (la plus stable)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(self.chroma_path)
            )
            
            self.logger.info(f"✅ Base sauvegardée: {self.chroma_path}")
            return vector_store
            
        except Exception as e:
            self.logger.error(f"❌ Erreur construction: {e}")
            return None

# ==========================
# PIPELINE PRINCIPAL
# ==========================
class PDFIngestionPipeline:
    """Pipeline spécialisé pour l'ingestion de livres PDF"""
    
    def __init__(self, data_path: str = "../data", chroma_path: str = "../chroma_db"):
        self.data_path = Path(data_path)
        self.chroma_path = Path(chroma_path)
        self.logger = setup_logging()
        
        self.pdf_loader = PDFLoaderStrategy(self.logger)
        self.processor = DocumentProcessor(self.logger)
        self.vector_builder = VectorStoreBuilder(chroma_path, self.logger)
    
    def run_ingestion(self) -> bool:
        """Exécute le pipeline complet"""
        self.logger.info("🚀 DÉMARRAGE INGESTION PDF")
        
        # 1. Trouver le PDF
        pdf_files = list(self.data_path.glob("*.pdf"))
        if not pdf_files:
            self.logger.error("❌ Aucun PDF trouvé")
            return False
        
        pdf_file = pdf_files[0]  # Prendre le premier PDF
        self.logger.info(f"📖 Traitement: {pdf_file.name}")
        
        # 2. Chargement du PDF
        documents = self.pdf_loader.load_pdf(pdf_file)
        if not documents:
            self.logger.error("❌ Échec chargement PDF")
            return False
        
        # 3. Découpage en chunks
        chunks = self.processor.create_chunks(documents)
        if not chunks:
            self.logger.error("❌ Échec création chunks")
            return False
        
        # 4. Construction base vectorielle
        vector_store = self.vector_builder.safe_build_vector_store(chunks)
        
        if vector_store:
            self.logger.info("🎉 INGESTION RÉUSSIE!")
            return True
        else:
            self.logger.error("💥 ÉCHEC CONSTRUCTION BASE")
            return False

# ==========================
# EXÉCUTION
# ==========================
if __name__ == "__main__":
    # Installation de PyMuPDF si nécessaire
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("📦 Installation de PyMuPDF...")
        os.system("pip install pymupdf")
    
    pipeline = PDFIngestionPipeline("../data", "../chroma_db")
    success = pipeline.run_ingestion()
    
    if success:
        pipeline.logger.info("✨ PRÊT POUR LA SUITE DU PROJET!")
    else:
        pipeline.logger.error("💥 VÉRIFIEZ LE FICHIER PDF")
        exit(1)