import PyPDF2
import streamlit as st
import os
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import anthropic
from langchain.llms import Anthropic
from langchain.chat_models import ChatAnthropic

#load_dotenv()
#anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
anthropic_api_key="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"

class Generator_Anthropic:
    def __init__(self):
        self.system_prompt = self.get_system_prompt()

        self.user_prompt = HumanMessagePromptTemplate.from_template("{question_input}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )

        self.chat = ChatAnthropic(
            anthropic_api_key = anthropic_api_key,
            max_tokens_to_sample = 5000,
            #streaming=True,
            #verbose=True,
            #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

        self.chain = LLMChain(
            llm=self.chat, 
            prompt=full_prompt_template,
            #memory: BaseMemory | None = None,
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
        Solution:
        """

        system_prompt = """
        Context: {context}, 
        Start you answer here:
        """

        system_prompt_old = """
        I upoaded pdf files here, and I may ask questions about the pdf files:
        PDF files: {context}, 
        Start you answer in {language} here:
        """
        return SystemMessagePromptTemplate.from_template(system_prompt)

    def run_chain(self, language, context, question):
        return self.chain.run(
            language=language, context=context, question_input=question
        )