# from langchain.vectorstores import Weaviate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings  # 导入新版本中的OPENAI类
from langchain.prompts import ChatPromptTemplate
from langmain import chunk_get
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain  # 用于构建RAG
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
from openai import OpenAI
# import weaviate
# from weaviate.embedded import EmbeddedOptions
# from langchain_openai import OpenAIEmbeddings  # 旧版本的openai类
# from langchain.embeddings import OpenAIEmbeddings

def AI(query,file_path):
    """此代码块用于存储和嵌入外部知识库产生的向量块"""
    """embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )"""
    chunks = chunk_get(file_path)
    persist_directory = "./chroma_langchain_db"# 存储到本地矢量库
    # 使用 Ollama 嵌入模型生成文档向量表示，并存储到 Chroma 向量数据库中
    vectorstore = Chroma.from_documents(persist_directory=persist_directory, documents=chunks, embedding=OllamaEmbeddings(model='llama3.1'))
    # vectorstore = FAISS.from_documents(documents=chunks, embedding=OllamaEmbeddings(model='llama3.1'))


    """weaviate向量库不支持windows"""
    """client = weaviate.Client(
        embedded_options=EmbeddedOptions()
    )
    # 向量存储
    vectorstore = Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=OllamaEmbeddings(model='llama3.1'),
        by_text=False
    )"""
    # 检索
    retriever = vectorstore.as_retriever()

    # 撰写合适的prompt模板，可结合用户输入
    template = """你是新时代的文档检索帮助助手，请使用以下检索到的上下文来回答这个问题。如果您不知道答案，就直接说不知道。请最多使用三句话，并保持回答简洁。
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
    """prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{input}"),
        ]
    )"""
    print(prompt)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    llm = ChatOllama(
        model="llama3.1",
        temperature=10,
        # other params...
    )
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    # question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    # query = "陈科海是谁?"
    responce = rag_chain.invoke(query)
    return responce
