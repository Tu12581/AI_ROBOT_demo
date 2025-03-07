import os
import langchain
import langchain_openai
import zhipuai
from langchain.text_splitter import CharacterTextSplitter
import requests
import dotenv
# 预加载环境，从.env中读取
dotenv.load_dotenv()
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader, WebBaseLoader, DirectoryLoader,FireCrawlLoader
# from langchain.document_loaders import  UnstructuredHTMLLoader
import bs4
"""此代码块用于配置环境以及处理文档"""
# 在env中加载配置并连接到langsmith
FIRECRAWL_API_KEY="fc-8715f26f57a9420eb35b72cbc17bcdb6"  #  用于firecrawl爬取网站，也可以用付费的spiderloader
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.47"
OPENAI_API_KEY = "sk-A4Djx7p09JQz3FBEF7RTTDnH2JZi972HHOKfE4qE7bX4CXOY"
LANGCHAIN_TRACING_V2 = 'true'
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "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"
LANGCHAIN_PROJECT = "pr-candid-gravel-78"

"""llm = ChatOpenAI()
llm.invoke("Hello, world!")"""

# 从指定的url上读取内容并作为文本保存
'''url = "https://baike.baidu.com/item/%E6%8B%9C%E4%BB%81%E6%85%95%E5%B0%BC%E9%BB%91%E8%B6%B3%E7%90%83%E4%BF%B1%E4%B9%90%E9%83%A8/4604932"
res = requests.get(url)
with open("temp.html", "w", encoding='utf-8') as f:
    f.write(res.text)'''

# html文档加载器
# loader = UnstructuredHTMLLoader("temp.html")
# 将网页中的所有文本加载到我们可以在下游使用的文档格式
"""loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)"""
# txt文档加载器
"""files = os.listdir('teacher')
files = ['teacher/' + file for file in files]
loader = TextLoader('1.txt', autodetect_encoding='utf-8')
documents = loader.load()  # 用langchain的textload来加载文本到输入中"""
# 批量处理文件夹中文件
"""loader = DirectoryLoader('teacher',glob="**/*.txt")
documents = loader.load()"""
# 分块处理文本的过程,分割为 500 个字符的块，并在块之间重叠 50 个字符
"""text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)"""

def chunk_get(file_path):
    # txt文档加载器
    loader = TextLoader(file_path, autodetect_encoding='utf-8')
    documents = loader.load()  # 用langchain的textload来加载文本到输入中
    # 批量处理文件夹中文件
    """loader = DirectoryLoader('teacher',glob="**/*.txt")
    documents = loader.load()"""
    # 分块处理文本的过程,分割为 500 个字符的块，并在块之间重叠 50 个字符
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def web_chunk_get(url,mode):  # 从网页爬取后处理文件
    # scrape: 抓取单个网址并返回markdown。
    # crawl: 爬取网址及所有可访问的子页面，并返回每个页面的markdown。
    if mode == "mode2":
        print(url[1])
        flag = True # 用于标记
        chunks = 1
        for temp_url in url[:5]:
            try:
                print("加载url为" + temp_url)
                loader = FireCrawlLoader(url=temp_url, mode="crawl")
                documents = loader.load()  # 用langchain的textload来加载文本到输入中
                flag = True
            except:
                flag = False
                chunks = None
            if flag:
                break
        # 批量处理文件夹中文件
        """loader = DirectoryLoader('teacher',glob="**/*.txt")
        documents = loader.load()"""
        for doc in documents:
            for md in doc.metadata:
                doc.metadata[md] = str(doc.metadata[md])
        print(documents[0].metadata)
        # 分块处理文本的过程,分割为 500 个字符的块，并在块之间重叠 50 个字符
        if chunks is not None:
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
            chunks = text_splitter.split_documents(documents)
        return chunks
    else:
        doc = []  # 存储每个网页爬取结果
        for temp_url in url:
            try:
                loader = FireCrawlLoader(url=temp_url, mode="crawl")
                documents = loader.load()  # 用langchain的textload来加载文本到输入中
                doc.append(documents)
                print(documents[0].metadata)
            except:
                flag = False
        # 分块处理文本的过程,分割为 500 个字符的块，并在块之间重叠 50 个字符
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = text_splitter.split_documents(documents)
        return chunks
# curl https://api.chatanywhere.tech/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-A4Djx7p09JQz3FBEF7RTTDnH2JZi972HHOKfE4qE7bX4CXOY' -d '{"model": "gpt-3.5-turbo","messages": [{"role": "user", "content": "Say this is a test!"}],"temperature": 0.7}'