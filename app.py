from PyPDF2 import PdfReader
import streamlit as st
import os
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    MessagesPlaceholder,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import anthropic
from langchain.llms import Anthropic
from langchain.chat_models import ChatAnthropic
from Generator_Anthropic import Generator_Anthropic
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

#load_dotenv()
#openai_api_key = os.getenv('OPENAI_API_KEY')
#print(api_key)
openai_api_key = "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"
def retrieve_pdf_text(pdf_file):
    text = ""
    try:
        # Using PyPDF2 for reading PDFs with the updated class.
        pdf_reader = PdfReader(pdf_file)
        for page_number in range(len(pdf_reader.pages)):  # Updated this line to get number of pages
            page = pdf_reader.pages[page_number]  # Updated this line to get a page
            page_text = page.extract_text()
            if page_text is not None:  
                text += page_text
            else:
                print(f"Warning: No text found on page {page_number + 1}")  
        #print(text)
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
    return text
class Generator:
    def __init__(self):
        self.system_prompt = self.get_system_prompt()

        self.user_prompt = HumanMessagePromptTemplate.from_template("{question_input}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )

        self.chat = ChatOpenAI(
            model_name = "gpt-4",
            temperature=0, 
            openai_api_key = openai_api_key,
            #streaming=True, 
            callbacks=[StreamingStdOutCallbackHandler()],
            #cache= True,
            #n = 2,
            #verbose: bool = _get_verbosity,
            #callbacks: Callbacks = None,
            #callback_manager: BaseCallbackManager | None = None,
            #tags: List[str] | None = None,
            #metadata: Dict[str, Any] | None = None,
            #client: Any,
            #model_kwargs: Dict[str, Any] = dict,
            #openai_api_base: str | None = None,
            #openai_organization: str | None = None,
            #openai_proxy: str | None = None,
            #request_timeout: float | Tuple[float, float] | None = None,
            #max_retries: int = 6,
            #streaming= True,
            #max_tokens: int | None = None,
            #tiktoken_model_name: str | None = None
            )

        self.chain = LLMChain(
            llm=self.chat, 
            prompt=full_prompt_template,
            #memory= self.memory,
            #callbacks: Callbacks = None,
            #callback_manager: BaseCallbackManager | None = None,
            #verbose: bool = _get_verbosity,
            #tags: List[str] | None = None,
            #metadata: Dict[str, Any] | None = None,
            #output_key: str = "text",
            #output_parser: BaseLLMOutputParser = NoOpOutputParser,
            #return_final_only: bool = True,
            #llm_kwargs: dict = dict
            )

    def get_system_prompt(self):
        # system_prompt_example not in use
        system_prompt_example = """
        The following is a friendly conversation between a human and an AI. 
        If the AI does not know the answer to a question, it tries its best to provide 
        as much relevant information as possible.
        {context}
        Instruction: 
        Based on the above documents, provide a detailed answer using {language}.
        Solution in {language}:
        """

        system_prompt_old = """
        I upoaded pdf files here, and I may ask questions about the pdf files:
        PDF files: {context}, 
        Start you answer in {language} here:
        """
        system_prompt = """
        Context: {context}, 
        Start you answer here:
        """
        return SystemMessagePromptTemplate.from_template(system_prompt)

    def run_chain(self, language, context, question):
        return self.chain.run(
            language=language, context=context, question_input=question,
        )
"""
# Create a Streamlit app
#st.set_page_config(layout="wide")
st.title("Prompt Playground🥳")

# Get the user's choice
model_choice = st.selectbox("Choose a model", ["Small Model", "Large Model"])

# Perform an action based on the user's choice
if model_choice == "Small Model":
    st.write("You chose OpenAI. Performing action with OpenAI model...")
    st.session_state.Generator = Generator()
elif model_choice == "Large Model":
    st.write("You chose Anthropic. Performing action with Anthropic model...")
    st.session_state.Generator = Generator_Anthropic()

# Initialize the conversation history if it doesn't exist.
if "history" not in st.session_state:
    st.session_state.history = []

if "context" not in st.session_state:
    st.session_state.context = ""
# create a upload file widget for a pdf
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# if pdf files are uploaded
if pdf_files:
    # retrieve the text from the pdfs
    texts = [retrieve_pdf_text(pdf_file) for pdf_file in pdf_files]
    
    # concatenate texts from all PDFs
    st.session_state.context = "\n\n".join(texts)
else:
    st.write("Please upload pdf files")

# create a button that clears the context
if st.button("Clear context"):
    st.session_state.context = ""

# If there's context, proceed.
language = st.selectbox("Language", ["English", "中文"])
if language != "English":
    st.session_state.language = "中文"
else:
    st.session_state.language = "english"

# Colors to be used in alternating manner for Q/A pairs
colors = ["#f0f8ff", "#faf0e6"]  # AliceBlue and OldLace color codes. You can choose your own.

for idx, interaction in enumerate(st.session_state.history):
    # Check if either question or answer is None or empty and handle accordingly
    question = interaction['question'] or "No Question Provided"
    answer = interaction['answer'] or "No Answer Available"
    
    # Choose color based on index (even/odd)
    color = colors[idx % len(colors)]
    
    # Using HTML and CSS for styling within markdown
    st.markdown(f"<div style='background-color: {color}; padding: 10px;'><b>Q{idx + 1}: {question}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: {color}; padding: 10px;'><b>A{idx + 1}: {answer}</b></div>", unsafe_allow_html=True)


# create a text input widget for a question
question = st.text_input("Input the generation prompt")

# create a button to run the model
if st.button("Run"):
    # run the model
    generator_response = st.session_state.Generator.run_chain(
        language=st.session_state.language, context=st.session_state.context, question=question
    )

    # Add the question and answer to the history.
    st.session_state.history.append({"question": question, "answer": generator_response})

    # refresh streamlit to display new response immediately
    st.experimental_rerun()

# create a button to clear the history
if st.button("Clear History"):
    if st.button("I want to Clear History"):
        st.session_state.history = []
        # refresh streamlit
        st.experimental_rerun()
"""
