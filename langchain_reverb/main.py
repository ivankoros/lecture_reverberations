import whisper
from dotenv import load_dotenv
load_dotenv()

model = whisper.load_model("medium")
result = model.transcribe('/home/koros/PycharmProjects/whisper_lecture/My recording 2.m4a',
                          verbose=True,
                          initial_prompt="This is a lecture at a university Biology Department."
                          )

print(result["text"])


from langchain.document_loaders import TextLoader

loader = TextLoader("My recording 2.txt")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

question = "What is an inbred species?"
docs = vectorstore.similarity_search(question)
print(len(docs))

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
print(qa_chain({"query": question}))