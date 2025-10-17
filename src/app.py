import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Ajouter le chemin parent pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import DataIngestor
from rag_chain import RAGChatbot

def initialize_session_state():
    """Initialise l'état de la session Streamlit"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot_initialized" not in st.session_state:
        st.session_state.chatbot_initialized = False
    if "vector_store_built" not in st.session_state:
        st.session_state.vector_store_built = False

def main():
    st.set_page_config(
        page_title="Chatbot RAG avec Gemini",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title(" Chatbot RAG avec Google Gemini")
    st.markdown("Posez des questions sur vos documents en utilisant l'IA de Google!")
    
    initialize_session_state()
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header(" Configuration")
        
        # Vérification de la clé API
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.success(" Clé API Gemini configurée")
        else:
            st.error(" Clé API manquante")
            st.info("Ajoutez GOOGLE_API_KEY dans le fichier .env")
        
        st.markdown("---")
        
        # Bouton pour construire la base vectorielle
        if st.button(" Construire la base de connaissances", type="primary"):
            with st.spinner("Construction de la base vectorielle..."):
                ingestor = DataIngestor()
                result = ingestor.run_ingestion()
                if result:
                    st.session_state.vector_store_built = True
                    st.success("Base de connaissances mise à jour!")
                    # Réinitialiser le chatbot pour prendre en compte la nouvelle base
                    st.session_state.chatbot_initialized = False
                else:
                    st.error("Échec de la construction. Vérifiez vos documents.")
        
        st.markdown("---")
        st.markdown("###  Documents dans data/")
        data_dir = "./data"
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            if files:
                for file in files:
                    st.write(f" {file}")
            else:
                st.write("Aucun document trouvé")
        else:
            st.write("Dossier data/ non trouvé")
    
    # Zone de chat principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Affichage de l'historique des messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Afficher les sources pour les réponses de l'assistant
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander(" Sources utilisées"):
                        for i, source in enumerate(message["sources"]):
                            source_content = source.page_content
                            source_name = os.path.basename(source.metadata.get('source', 'Inconnu'))
                            
                            st.markdown(f"**Source {i+1}** (de {source_name}):")
                            st.info(source_content[:400] + "..." if len(source_content) > 400 else source_content)
                            st.markdown("---")
    
    with col2:
        st.markdown("###  Conseils")
        st.info("""
        - Posez des questions précises
        - Le système utilise vos documents
        - Cliquez sur "Sources" pour voir les références
        """)
        
        # Statut du système
        st.markdown("###  Statut")
        if st.session_state.vector_store_built:
            st.success(" Base vectorielle prête")
        else:
            st.warning(" Base vectorielle à construire")
        
        if st.session_state.chatbot_initialized:
            st.success(" Chatbot initialisé")
        else:
            st.warning(" Chatbot non initialisé")
    
    # Input utilisateur
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajouter le message utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Initialiser le chatbot si nécessaire
        if not st.session_state.chatbot_initialized:
            try:
                with st.spinner("Initialisation du chatbot Gemini..."):
                    st.session_state.chatbot = RAGChatbot()
                    st.session_state.chatbot_initialized = True
            except Exception as e:
                st.error(f" Erreur d'initialisation: {e}")
                return
        
        # Générer la réponse
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents..."):
                try:
                    response = st.session_state.chatbot.ask_question(prompt)
                    
                    st.markdown(response["answer"])
                    
                    # Ajouter la réponse à l'historique avec les sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Erreur: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()