# from langchain.vectorstores import Weaviate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings  # 导入新版本中的OPENAI类
from langchain.prompts import ChatPromptTemplate
from langmain import chunks
from openai import OpenAI
# import weaviate
# from weaviate.embedded import EmbeddedOptions
# from langchain_openai import OpenAIEmbeddings  # 旧版本的openai类
# from langchain.embeddings import OpenAIEmbeddings

"""此代码块用于存储和嵌入外部知识库产生的向量块"""
"""embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)"""
# 使用 OpenAI 嵌入模型生成文档向量表示，并存储到 Chroma 向量数据库中
vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())

"""weaviate向量库不支持windows"""
"""client = weaviate.Client(
    embedded_options=EmbeddedOptions()
)
# 向量存储
vectorstore = Weaviate.from_documents(
    client=client,
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    by_text=False
)"""
# 检索
retriever = vectorstore.as_retriever()

# 撰写合适的prompt模板，可结合用户输入
template = """你是一个跨越时代的问答任务助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。最多使用三个句子，保持答案简洁。
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
"""client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-A4Djx7p09JQz3FBEF7RTTDnH2JZi972HHOKfE4qE7bX4CXOY",
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.org/v1"
)"""
print(prompt)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)
query = "陈科海是谁?"
responce = rag_chain.invoke(query)
print(responce)


def AI(query):
    # query = "陈科海是谁?"
    responce = rag_chain.invoke(query)
    return responce
