import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers

# Function to preprocess PDFs and create vector stores
def preprocess_pdfs(directory_path):
    text = ""
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        pdf_reader = PdfReader(pdf_path)

        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    print('Chunks done')
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

# Function to load the LLM model
def load_llm():
    llm = CTransformers(
        model=r'F:\JMM\Langchain\ChatMedical\medicalBot\medical_bot\llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Function to create and get conversations
def get_conversations(vectorstore):
    llm =  load_llm()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversational_chain

# handle user question
def handleQuestion(user_question, conversations):
    print('HandleQuestion=',user_question)
    if st.session_state.conversations is not None:
    # ry:t
        response =conversations({'question': user_question})
        chat_history = response['chat_history']
    # except Exception as e:
    #     st.write("An error occurred:", str(e))
    #     chat_history = []  # Handle the error by setting an empty chat history

        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)




# Main function
def main():
    # Load API keys
    load_dotenv()

    # Set page layout
    st.set_page_config(page_title="Medical Chatbot", page_icon="👨‍⚕️")

    # Add CSS
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversations" not in st.session_state:
        st.session_state.conversations = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore =None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history =None

    # Set header
    st.header("Ask Anything About Medicine from Medical Chatbot :doctor")

    # Path to the folder containing PDFs
    pdf_folder_path = r'F:\JMM\Langchain\ChatMedical\medicalBot\medical_bot\dataset'

    # Check if preprocessing is done and cached
    if st.session_state.conversations is None:
        st.session_state.vectorstore = preprocess_pdfs(pdf_folder_path)
        print('Preprocessing done')
        st.session_state.conversations = get_conversations(st.session_state.vectorstore)
    print('embeddings Done')
    # User input
    user_question = st.text_input("Ask a question about Medicine")

    print(user_question)

    if user_question:
        handleQuestion(user_question, st.session_state.conversations)
        print('Im in main after handle question----')
        print('good')
        
# Inside your main function
if __name__ == "__main__":
    main()
else:
    st.write("This script is being imported. It should be executed as the main script.")

