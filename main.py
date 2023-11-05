import os
from flask import Flask, request
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, \
    ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import  OpenAIEmbeddings
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import keys
os.environ["OPENAI_API_KEY"] = keys.OPENAI

from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

api_args = reqparse.RequestParser()
api_args.add_argument("query",type=str,help="Input question(str) for the bot is required",required=True)
print("APPLICATION AND SERVER STARTED")





class ChatBot(Resource):
    print("CHATBOT MAIN CLASS STARTED")
    url = "https://en.wspa.pl/"
    urls = []
    loader = RecursiveUrlLoader(
        url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = loader.load()
    for link in docs:
        urls.append(link.metadata["source"])
        print(link.metadata["source"])
    llm = ChatOpenAI()

    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)


    def assistant(self,query):
        print("ASSISTANT STARTED")
        doks = self.vectorStore_openAI.similarity_search(query)
        doks_page_content = " ".join([d.page_content for d in doks])
        template = """
                You are a helpful assistant that can answer question a website based on the website's text content: {docs}
                Only use the factual information from the text to answer the question.
                If you don't have enough information to answer the question, say "This information is not available on the website, please contact email:mchoga@icloud.com"
                Your answers should be verbose and detailed
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "Answer the following question: {question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        response = chain.run(question=query, docs=doks_page_content)
        response = response.replace("\n", "")
        return response


    def get(self):
        print("GET CALLED")
        args = api_args.parse_args()
        body = request.get_json()

        return {"Answer ": self.assistant(body["query"])}


api.add_resource(ChatBot, "/api/chatbot")

if __name__ =="__main__":
    app.run(debug=True)













#
# loader = RecursiveUrlLoader(
#     url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
# )
# docs = loader.load()
# for link in docs:
#     urls.append(link.metadata["source"])
#     print(link.metadata["source"])
#
#
#
#
# llm = ChatOpenAI()
#
# loaders = UnstructuredURLLoader(urls=urls)
# data = loaders.load()
# text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
# docs = text_splitter.split_documents(data)
#
# embeddings = OpenAIEmbeddings()
#
#
#
#
# embeddings = OpenAIEmbeddings()
# vectorStore_openAI = FAISS.from_documents(docs, embeddings)
#
#
# def assistant(query):
#     doks = vectorStore_openAI.similarity_search(query)
#     doks_page_content = " ".join([d.page_content for d in doks])
#     template = """
#             You are a helpful assistant that can answer question a website based on the website's text content: {docs}
#             Only use the factual information from the text to answer the question.
#             If you don't have enough information to answer the question, say "This information is not available on the website, please contact email:mchoga@icloud.com"
#             Your answers should be verbose and detailed
#     """
#     system_message_prompt = SystemMessagePromptTemplate.from_template(template)
#     human_template = "Answer the following question: {question}"
#     human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
#     chat_prompt = ChatPromptTemplate.from_messages(
#         [system_message_prompt, human_message_prompt]
#     )
#     chain = LLMChain(llm=llm, prompt=chat_prompt)
#     response = chain.run(question=query, docs=doks_page_content)
#     response = response.replace("\n", "")
#     print(response)






# with open("faiss_store_openai.pkl","wb") as f:
#     pickle.dump(vectorStore_openAI,f)
#
#
# with open("faiss_store_openai.pkl","rb") as f:
#     vectorstore = pickle.load(f)
#



# chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=doks.as_retriever())







