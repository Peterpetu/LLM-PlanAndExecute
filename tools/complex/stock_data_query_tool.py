import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Weaviate
from langchain.agents import Tool
import weaviate

load_dotenv()

OPENAI_API_KEY = "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"
WEAVIATE_URL = "http://localhost:8080"

class StockDataQueryTool:
    def __init__(self, name="IBM Stock Data Query", description="Queries IBM stock data from a vector database"):
        self.name = name
        self.description = description
        self.module = "StockDataQuery"
        memory = ConversationBufferMemory(output_key="result")
        self.memory=memory

    def execute(self, query):
        client = weaviate.Client(WEAVIATE_URL)
        embed = OpenAIEmbeddings()
        vectorstore = Weaviate(client, "StockData", "combined", embedding=embed)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",
                     openai_api_key=OPENAI_API_KEY,
                     temperature=0,
                     max_tokens=1000)

        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2,
                                                                                                      "score_threshold": 0.20})

        template = """You are a helpful reviewer. You have access to vector database that has IBM stock data, you will get a question about the IBM stock from another AI agent. Answer the question.
        Please make you answer very consice but fit to the query, avoid respoding with too much informatin. Your answer should be short and precise.

        
        {context}
        {question}
        """

        qa = RetrievalQA.from_chain_type(llm, chain_type="stuff",
                                         retriever=retriever,
                                         memory=self.memory,
                                         chain_type_kwargs={
                                             "prompt": PromptTemplate(
                                                 template=template,
                                                 input_variables=["question", "context"],
                                             )},
                                         return_source_documents=False)
        response = qa({"query": query})
        
        return response

    def to_tool(self):
        return Tool(
            name=self.name,
            func=self.execute,
            description=self.description
        )
