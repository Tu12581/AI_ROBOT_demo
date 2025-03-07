基于RAG技术和langchain开发的检索增强助手
==
使用的模型为8Bllama3.1本地模型，设计了五个功能接口，分别是结合现有文档检索，利用firecrawl的http检索，自行设计爬虫的http检索，利用gpt3.5的API检索以及本地向量库检索<br>

①	结合现有文档检索
--
使用基础的rag链过程，用户提供文本文档和问题，根据文本文档以及问题生成用户回答。<br>

②	利用firecrawl的http检索
--
使用百度的url以及提供的问题从百度搜索引擎上爬取关于用户回答的前两页网页结果的所有链接,利用langchain提供的firecrawl加载器爬取这些网页上的内容，由于firecrawl设置的API额度有限，出于成本考虑我们不得不考虑仅爬取第一个可以成功访问的网站结果用于构建知识库，随后通过同样的流程构建rag链并对llm进行提问。<br>
 
③	自行设计爬虫的http检索
--
方法②虽然能绕开网站的反爬机制爬取内容，但是内容的质量无法允许我们筛选，因此我们自行设计了对百度网页内容的爬取方法，忽略掉返回值为空的网页内容和访问超时的网站，返回剩下的网站url和对应网站内容<br>
通过idf_tf方法从这些url的返回结果中筛选与问题相关度最高的前三个网页，利用这三个网页的内容来构建知识库并进行RAG结合的检索。<br>
 
④	利用gpt3.5的API检索
--
最开始构建RAG链使用的llm为openai3.5的API接口，但是我们发现了一个严重的问题，该接口的最大tokens限制仅为4096个tokens，这对于需要构建知识库的我们来说显然是远远不够的，然而本地的8B大模型由于算力有限性能往往不佳因此我们提出了两种方案:<br>
（1）	使用本地的大模型基于问题压缩输入的文本tokens，限制在4096个tokens以内，随后再将结果作为新的知识库喂给gpt3.5的API，然后返回结果。<br>
（2）	切割用户提供的文本档案，分批次喂给gpt3.5的API并压缩，随后再进行提问和解答。<br>
最后实现了第一种方案，第二种方案由于没有想到非常好的切割方法，因为我们无法很好地保证切割段落的完整性，这无疑会大幅影响gpt的性能，最终没有实现。<br>
 
⑤	本地向量库检索
--
将基于文档生成的向量的内容存储到了本地，因此完全可以根据本地向量知识库直接建立RAG链，这样就避免了需要反复提供文本或联网搜索的麻烦。<br>
然而该方法存在一定问题，倘若用户反复联网搜索或提供相同问答，会导致向量知识库的严重冗余，这不仅会占用过多内容，也会导致检索器查询的困难，更重要的是，由于检索器的查询方案为嵌入向量的相似度，于是很容易产生答非所问的情况，因为query很可能和关系没有那么大的其他内容相似度更好，例如提问：原神中甘雨的毕业武器是什么，由于原神一词的存在，这往往会导致该词向量的相似度和其他含原神更多的本地知识库相似度更好，最后导致答非所问。我们目前没有很好的办法取解锁向量知识库被污染的问题，目前的思路是改写检索器中的检索函数，可以尝试用idf-tf来减少如上述情况导致的误检索。<br>

嵌入向量的优化<br>
后续构建向量知识库会在考虑不同模型向量结构相同的情况下使用标记窗口更大，性能更好的嵌入模型如nomic-embed-text，或接入deepseek8B模型。<br>

若要修改使用的本地模型，请在history_vector文件中修改，请注意，嵌入向量数据库使用的模型最好与后续构建RAG链使用的模型一致，否则可能产生上述嵌入向量优化中提到的模型产生vector结构不同导致的冲突。<br>
