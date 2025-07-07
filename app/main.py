# conda activate travel
# cd "C:\Users\Sude\Desktop\Smart_Travel_Advisor"
# streamlit run app/main.py


import streamlit as st
from rag_utils import TravelAssistant
import os
from dotenv import load_dotenv

def main():
    # Initialize environment variables
    load_dotenv()

    # App title and description - moved inside main function
    st.title("🌍 Smart Travel Advisor")
    st.caption("Ask about destinations or predict flight prices")

    @st.cache_resource
    def init_assistant():
        """Initialize and cache the TravelAssistant instance"""
        try:
            assistant = TravelAssistant()
            return assistant
        except Exception as e:
            st.error(f"Failed to initialize assistant: {str(e)}")
            return None

    # Initialize assistant
    assistant = init_assistant()

    # Sidebar for system controls - moved after assistant initialization
    with st.sidebar:
        st.header("System Status")
        
        # Display system status indicators
        if assistant:
            rag_status = "Ready" if hasattr(assistant, 'index') and assistant.index else "Not loaded"
            llm_status = "Ready" if hasattr(assistant, 'client') and assistant.client else "Unavailable"
            model_status = "Ready" if hasattr(assistant, 'price_model') and assistant.price_model else "Disabled"
            status = [
                f"✓ RAG System: {rag_status}",
                f"✓ LLM Service: {llm_status}",
                f"✓ Price Predictor: {model_status}"
            ]
            if rag_status == "Not loaded":
                status.append("\nℹ️ To enable RAG, add PDFs to app/data/travel_docs/")
            st.info("\n".join(status))
        
        # Knowledge base reload button
        if st.button("🔁 Reload Knowledge Base"):
            with st.spinner("Rebuilding knowledge base..."):
                try:
                    assistant._init_rag_system()
                    st.success("Knowledge base reloaded!")
                    st.rerun()  # Refresh to show updated status
                except Exception as e:
                    st.error(f"Reload failed: {str(e)}")

    # Initialize chat messages if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your travel assistant. Ask me about destinations or flight prices."}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("E.g., 'Best time to visit India?' or 'Flight price from Delhi to Mumbai'"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if not assistant:
                st.error("Travel Assistant is not properly initialized. Please check system status.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        response = assistant.process_query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"⚠️ Sorry, I encountered an error: {str(e)}"
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()