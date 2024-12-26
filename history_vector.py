from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings  # 导入新版本中的OPENAI类
from langchain.prompts import ChatPromptTemplate
from langmain import chunk_get,web_chunk_get
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain,create_history_aware_retriever  # 用于构建RAG以及历史感知
from langchain.chains.combine_documents import create_stuff_documents_chain
from search import get_llm_url,get_text
from tf import idf_tf
from langchain_community.vectorstores import FAISS
# from langchain.vectorstores import Weaviate
# from langchain_ollama import OllamaEmbeddings
# import weaviate
# from weaviate.embedded import EmbeddedOptions
# from langchain_openai import OpenAIEmbeddings  # 旧版本的openai类
# from langchain.embeddings import OpenAIEmbeddings
"""次文件用于检索向量生成以及模型调用"""
def his_AI(query,file_path,mode):
    """此代码块用于存储和嵌入外部知识库产生的向量块"""
    """embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )"""
    # 存储到本地矢量库
    persist_directory = "./chroma_langchain_db"
    if mode == "mode1" or mode == "mode4":
        # 文档切割向量化与
        if file_path is None:
            chunks = None
            # vectorstore = Chroma.asimilarity_search(vectorstore=vectorstore, query=query)
        else:
            chunks = chunk_get(file_path)
            print(chunks)
    # 使用手动爬虫的情况
    elif mode == "mode3":
        texts = get_text(query)
        if len(texts) > 3:  # 当大于3时检索相关度最高的网页，返回结果
            texts = idf_tf(texts,query)
        print(texts)
        num = 0
        temp = ''
        for i in texts.values():
            num += 1
            temp += f"{num}、" + i + '\n'
        with open('temp.txt', 'w', encoding='utf-8') as f:
            f.write(temp)
            f.close()
        chunks = chunk_get("temp.txt")
        print(chunks)
    elif mode == "mode2":
        url = get_llm_url(query)
        chunks = web_chunk_get(url,mode)
        print(chunks)
    else:
        chunks = 1
    # 用户上传了文件的情况
    if chunks:
        """使用 Ollama 嵌入模型生成文档向量表示，并存储到 Chroma 向量数据库中
        nomic-embed-text: 一个高性能开放嵌入模型，具有较大的标记上下文窗口。
        vectorstore = Chroma.from_documents(documents=chunks,embedding=OllamaEmbeddings(model='nomic-embed-text'))"""
        if mode == "mode1" or mode == "mode4" or mode == "mode3":
            vectorstore = Chroma.from_documents(persist_directory=persist_directory,documents=chunks,
                                 embedding=OllamaEmbeddings(model='llama3.1'))
            vectorstore = Chroma.from_documents(documents=chunks,embedding=OllamaEmbeddings(model='llama3.1'))
            # 加载本地向量
            # 创建和存储本地向量库
            """vectorstore = Chroma.from_documents(persist_directory=persist_directory, documents=chunks,
                                            embedding=OllamaEmbeddings(model='llama3.1'))
               vectorstore.persist()"""
            # vectorstore.add_documents(documents=chunks,persist_directory=persist_directory,embedding=OllamaEmbeddings(model='llama3.1'))
            print("创建完毕")
        elif mode == "mode5":
            # 加载本地向量库
            vectorstore = Chroma(persist_directory=persist_directory,embedding_function=OllamaEmbeddings(model='llama3.1'))
            print(vectorstore)
            # 返回相似度最高的两个文档向量
            # 生成用于查询的嵌入向量
            # vectorstore = vectorstore.similarity_search(query,k=3)
        else:  # chroma不支持fire爬取后loader返回的chunks
            vectorstore = FAISS.from_documents(documents=chunks, embedding=OllamaEmbeddings(model='llama3.1'))

        # prompt写法废案1
        """system_prompt = (
            "你是新时代的文档检索帮助助手，请使用以下检索到的上下文来回答这个问题。如果您不知道答案，就直接说不知道。请最多使用三句话，并保持回答简洁。 "
            "\n\n"
            "{context}"
        )"""
        # prompt = ChatPromptTemplate.from_template(template)
        """client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key="sk-A4Djx7p09JQz3FBEF7RTTDnH2JZi972HHOKfE4qE7bX4CXOY",
            base_url="https://api.chatanywhere.tech/v1"
            # base_url="https://api.chatanywhere.org/v1"
        )"""
        # prompt写法废案1
        """prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )"""
        # 检索向量生成
        retriever = vectorstore.as_retriever()
        # 优化历史查询记录
        history_template = """
                给定一个聊天历史记录和用户的最新问题，这个问题可能会参考聊天历史中的上下文，
                制定一个独立的问题，这个问题在没有聊天历史的情况下也能理解。
                不要回答问题，只需要在需要时重新构建它，否则原样返回。
                Chat_history: {chat_history} 
                Input: {input} 
                Answer:
            """
        # 撰写合适的prompt模板，可结合用户输入
        # chat_history and input都是用于后续的输入设置的框
        if mode == "mode1" or mode == 'mode3':
            template = """你是新时代的文档检索帮助助手，请使用以下检索到的上下文来回答这个问题。如果您不知道答案，就直接说不知道，不要因此随意拼凑答案，请保持回答简洁。
                Context: {context} 
                Chat_history: {chat_history} 
                Input: {input} 
                Answer:
                """
        else:
            template = """你是新时代的文档检索帮助助手，请你总结获取到的上下文和信息。
                            Context: {context} 
                            Chat_history: {chat_history} 
                            Input: {input} 
                            Answer:
                            """
        # 生成prompt和历史查询prompt
        prompt = ChatPromptTemplate.from_template(template)
        history_prompt = ChatPromptTemplate.from_template(history_template)
        print(prompt)
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        llm = ChatOllama(
            model="llama3.1",
            temperature=0
            # other params...
        )
        # 根据检索调用create_history_aware_retriever函数构建历史感知检索
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, history_prompt)
        question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        # query = "陈科海是谁?"
        # 调用API，先总结，再放入API嵌入
        if mode == 'mode4':
            responce = rag_chain.invoke({"input": "请你帮我将这段文字总结一下，总结在4000个tokens以内", "chat_history": question_answer_chain})
            with open('temp.txt', 'w',encoding='utf-8') as f:
                f.write(responce["answer"])
            chunks = chunk_get("temp.txt")
            vectorstore = Chroma.from_documents(documents=chunks,
                                                embedding=OllamaEmbeddings(model='llama3.1'))
            retriever = vectorstore.as_retriever()
            template = """你是新时代的文档检索帮助助手，请使用以下检索到的上下文来回答这个问题。如果您不知道答案，就直接说不知道。请最多使用三句话，并保持回答简洁。
                            Question: {question} 
                            Context: {context} 
                            Answer:
                            """
            # 生成prompt和历史查询prompt
            prompt = ChatPromptTemplate.from_template(template)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
            )
            responce = rag_chain.invoke(query)
            return responce
        else:
            responce = rag_chain.invoke({"input": query, "chat_history": question_answer_chain})
            print(responce)
            # 手动爬虫，加入信息获取的网址
            if mode == 'mode3':  # 手动爬虫的情况
                td = "答案查询网址为"
                if texts:
                    for i in texts.keys():
                        td = td + i + ',\n'
                    td = '\n' + td
                    return responce["answer"] + td
                else:
                    responce = '网页检索失败，请检查您的网络'
                    return responce
            else:
                return responce["answer"]
    # 用户未上传文件的情况
    else:
        if mode == 'mode1':  # 直接调用API进行检索
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            responce = llm.invoke(query)
            llm = ChatOllama(
                model="llama3.1",
                temperature=0
                # other params...
            )
            responce = llm.invoke(query)
        else:
            responce = '网页检索失败，请检查您的网络'
        return responce
