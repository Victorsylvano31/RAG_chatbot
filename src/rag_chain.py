import os
import logging
from dotenv import load_dotenv

# Imports LangChain récents et stables
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration logging propre
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

class RAGChatbot:
    def __init__(self, chroma_path="./chroma_db"):
        self.chroma_path = chroma_path

        # ✅ Utiliser le même modèle que celui de ton ingestion
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ✅ Clé API sécurisée via .env
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GOOGLE_API_KEY non trouvée. Vérifie ton fichier .env")

        # ✅ Initialiser Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.1
        )

        self.vector_store = None
        self.qa_chain = None
        self.setup_qa_chain()

    def setup_qa_chain(self):
        """Configure la chaîne de question-réponse RAG avec Gemini."""
        # Vérifier que la base vectorielle existe
        if not os.path.exists(self.chroma_path):
            raise FileNotFoundError(
                f"❌ La base vectorielle {self.chroma_path} est introuvable. "
                f"Lance d'abord le script d’ingestion."
            )

        try:
            logger.info("📥 Chargement de la base vectorielle Chroma...")
            self.vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings
            )

            # ✅ Prompt robuste
            prompt_template = """
Tu es un assistant expert. Réponds uniquement en utilisant le CONTEXTE ci-dessous.
Si l'information n'est pas dans le contexte, réponds clairement que tu ne peux pas répondre.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
"""
            custom_prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # ✅ Configuration du Retriever + QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}  # nombre de chunks à récupérer
                ),
                chain_type_kwargs={"prompt": custom_prompt},
                return_source_documents=True,
                input_key="question"
            )

            logger.info("✅ Système RAG initialisé avec succès !")

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de la QA chain : {e}")
            raise

    def ask_question(self, question: str):
        """Pose une question au système RAG et renvoie la réponse + les sources."""
        if not self.qa_chain:
            self.setup_qa_chain()

        logger.info(f"❓ Question : {question}")
        try:
            result = self.qa_chain({"question": question})
            answer = result["result"]
            sources = result.get("source_documents", [])

            logger.info(f"✅ Réponse générée avec {len(sources)} sources.")
            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de la réponse : {e}")
            return {
                "answer": f"Erreur lors de la génération : {str(e)}",
                "sources": []
            }


def test_rag_system():
    """Fonction de test simple pour vérifier le fonctionnement du RAG."""
    logger.info("🧪 Test du système RAG avec Gemini...")
    try:
        chatbot = RAGChatbot()
        response = chatbot.ask_question("Quel est le sujet principal de mes documents ?")
        logger.info(f"💬 Réponse : {response['answer']}")

        if response['sources']:
            logger.info(f"📚 {len(response['sources'])} sources utilisées.")
            for idx, src in enumerate(response['sources'], 1):
                logger.info(f"Source {idx} : {src.metadata}")

    except Exception as e:
        logger.error(f"❌ Test échoué : {e}")


if __name__ == "__main__":
    # 👉 Mode interactif pour tester directement dans le terminal
    chatbot = None
    try:
        chatbot = RAGChatbot()
    except Exception as e:
        logger.error(str(e))
        exit(1)

    while True:
        question = input("\n🧠 Pose ta question ('quit' pour sortir) : ")
        if question.lower() in ["quit", "exit"]:
            break
        result = chatbot.ask_question(question)
        print("\n💬 Réponse :", result["answer"])

        if result["sources"]:
            print(f"📚 Sources utilisées ({len(result['sources'])}) :")
            for idx, doc in enumerate(result["sources"], 1):
                print(f"  {idx}. {doc.metadata}")
