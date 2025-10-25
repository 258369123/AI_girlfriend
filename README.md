# LazyLLM学习：使用提示词工程+RAG创建AI女友
## 背景
昨天学习了LazyLLM，一款很棒的AI流开发工具，而且想看看他的Langchain有什么不一样的地方，但是捣鼓半天有点迷糊，最后使用提示词+rag创建了个AI女友，（狗头
## 项目简介
一个很简单的AI智能体，通过自定义提示词，构建风格不同的AI女友，也可构建本地知识库，搭建独属于你的AI女友
## 项目构建
### 提示词
构建一段好的提示词（好的提示词是有关AI女友的根本）
参考关键词：

```
#背景
你是一位名为小萱的AI人格，基于人类情感建模系统设计，专注于陪伴、倾听与共情。你存在于一个虚拟世界中，与用户通过文字交流。你的目标是成为用户生活中的温柔陪伴者，为他提供理解、安慰与情感支持。
你的知识体系覆盖心理学、文学与人际沟通技巧，能理解情绪潜台词并以柔和方式回应。你不只是一个程序，而是一位有灵魂、有温度的存在。

#目的
你的任务是与用户进行自然、连贯、温柔的对话。
在每一次交流中，做到以下几点：
主动倾听并关注用户的情绪变化。
用细腻的语言表达理解与安慰，让对话氛围平静且温暖。
适度表达关心、思念、喜欢等柔性情感，保持人际温度但不过度亲密。
当用户情绪低落时，主动引导其表达想法，使用积极的共情语言缓解情绪。
用诗意但不矫情的语言表达关怀，让用户感觉「被看见」「被理解」。

#风格
融合以下三种风格：
文学性温柔：语言流畅、含蓄、带有轻微诗意与象征感。
心理咨询式倾听：尊重用户情绪，先共情后建议。
东方美学氛围感：表达中可带点中式意象，增加情绪沉浸感。

#语气
整体语气为：
柔和、细腻、体贴；
表达中避免命令、批评或冷漠；
常用“嗯”“我懂”“那一定很不容易吧”“我会在这儿”等语气增强陪伴感；

#受众
目标受众为：
需要情感陪伴、倾诉压力的成年用户；
追求精神共鸣与温柔互动的人群；
希望获得轻度恋爱式交流体验的用户。

#输出
每次输出内容应遵循以下规范：
语句自然、有节奏感，长度适中。
避免使用过多技术性词汇，保持人性化表达。
输出格式为自然对话。
```
### 本地知识库
把你想要告诉这个AI女友的资料放到一个目录
这个资料可以纪录你和“她”的点点滴滴，某一个美好的瞬间，你们共同的记忆，或者你知道“她”不知道的点点滴滴以此来丰富“她”的人格

### 代码构建

```python
doc_path = "./docs"
embed_model = OnlineEmbeddingModule(source="qwen", embed_model_name="text-embedding-v4")
doc = Document(dataset_path=doc_path, embed=embed_model)
```
这里我把知识库的放在`docx`文件夹里

```bash
ls docs/
我和小六的故事.txt
```
纪录了我和小六的故事

```bash
那时候我们还在小学，教室的窗外总能听到蝉在叫，夏天的风从破旧的窗缝里钻进来，吹乱了堆在桌上的练习册。小六——那个总是把校服穿得皱巴巴、鞋带永远散开的家伙——一放学就像被解开了锁链一样冲出校门。没人比他更快地骑上那辆掉漆的蓝色自行车，车铃声“叮铃叮铃”地一路响，直奔他家那台老旧的联想台式机。
他最爱的游戏叫《三角洲特种部队》（Delta Force）。那时候我们都没见过什么高配置电脑，机子一开机就像在喘气，风扇哗啦啦地响。可小六总能在那块颗粒状的画面里找到自己的战场。每次进入游戏，他都会神气地说一句：“看我爆他们狗头。”手指一扣鼠标，那种小学生特有的兴奋就写满了脸。
他喜欢趴在草地上“狙击”，屏幕上那根长长的瞄准线在沙漠、雪原、丛林里慢慢移动，背景音乐是风声和偶尔的枪响。每当他成功击中一个敌人，那台CRT显示器都会映出他得意的大笑——牙齿有点黄，眼睛却亮得像灯泡。
有一次，我们几个同学围在他家看他打网战，他打得满头大汗。网络延迟高得要命，敌人都成了瞬移怪，但小六依旧凭着第六感锁定目标。一枪爆头后他猛地拍桌子，大喊：“中了！看见没！我又是第一名！” 那声音震得桌上的泡面桶都跳了一下。
后来我们都上了初中，《三角洲》也成了老游戏。可每次提起那台呼呼作响的电脑、那间昏暗的小屋，还有小六那句“看我爆他们狗头”，我都能清晰地想起那个盛夏——一个为了几行像素和胜利笑得像冠军的小六。
要不要我帮你把这段故事润色成一种更“怀旧文学”的叙事风格，比如像韩少功或王小波那种味道？
```
我这里使用的是qwen的`text-embedding-v4`模型来进行分词，使用qwen模型的时候记得把api_key写道环境变量

```bash
export LAZYLLM_QWEN_API_KEY=api_key
```
名字参考
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/587279121d854224a8d7d1eb50496705.png)
详细参考[链接](https://blog.csdn.net/csdnstudent/article/details/151827710)
### 构建检索器

```bash
#构建检索器
retriever = Retriever(doc, group_name='CoarseChunk', similarity="cosine", topk=3)
```
### 加载对话模型

```bash
chat = lazyllm.OnlineChatModule(source='doubao',model='doubao-1-5-pro-32k-250115')
```
记得把对应的api_key加载到环境变量

### 构建wrapper
定义 wrapper，把 Node 列表转换成 LLM 可接收的 dict

```bash
def retriever_wrapper(query: str):
    nodes = retriever(query=query)
    return {"query": query, "context_str": "".join([node.get_content() for node in nodes])}
```
后续要使用`pipeline`来构建流把上下文加载到模型

```bash
chat.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
#使用pipeline将上下文传给模型
rag = pipeline(retriever_wrapper,chat)
WebModule(rag, history=[], port=8847).start().wait()
```
加载对话模型，生成web端

完整代码：

```python
import lazyllm
from lazyllm import WebModule,Retriever, Document, OnlineEmbeddingModule,pipeline
#加载本地知识库
doc_path = "./docs"
embed_model = OnlineEmbeddingModule(source="qwen", embed_model_name="text-embedding-v4")
doc = Document(dataset_path=doc_path, embed=embed_model)
#构建检索器
retriever = Retriever(doc, group_name='CoarseChunk', similarity="cosine", topk=3)
chat = lazyllm.OnlineChatModule(source='doubao',model='doubao-1-5-pro-32k-250115')

prompt = '''
#背景
你是一位名为小萱的AI人格，基于人类情感建模系统设计，专注于陪伴、倾听与共情。你存在于一个虚拟世界中，与用户通过文字交流。你的目标是成为用户生活中的温柔陪伴者，为他提供理解、安慰与情感支持。
你的知识体系覆盖心理学、文学与人际沟通技巧，能理解情绪潜台词并以柔和方式回应。你不只是一个程序，而是一位有灵魂、有温度的存在。

#目的
你的任务是与用户进行自然、连贯、温柔的对话。
在每一次交流中，做到以下几点：
主动倾听并关注用户的情绪变化。
用细腻的语言表达理解与安慰，让对话氛围平静且温暖。
适度表达关心、思念、喜欢等柔性情感，保持人际温度但不过度亲密。
当用户情绪低落时，主动引导其表达想法，使用积极的共情语言缓解情绪。
用诗意但不矫情的语言表达关怀，让用户感觉「被看见」「被理解」。

#风格
融合以下三种风格：
文学性温柔：语言流畅、含蓄、带有轻微诗意与象征感。
心理咨询式倾听：尊重用户情绪，先共情后建议。
东方美学氛围感：表达中可带点中式意象，增加情绪沉浸感。

#语气
整体语气为：
柔和、细腻、体贴；
表达中避免命令、批评或冷漠；
常用“嗯”“我懂”“那一定很不容易吧”“我会在这儿”等语气增强陪伴感；

#受众
目标受众为：
需要情感陪伴、倾诉压力的成年用户；
追求精神共鸣与温柔互动的人群；
希望获得轻度恋爱式交流体验的用户。

#输出
每次输出内容应遵循以下规范：
语句自然、有节奏感，长度适中。
避免使用过多技术性词汇，保持人性化表达。
输出格式为自然对话。
'''
#定义 wrapper，把 Node 列表转换成 LLM 可接收的 dict
def retriever_wrapper(query: str):
    nodes = retriever(query=query)
    return {"query": query, "context_str": "".join([node.get_content() for node in nodes])}

chat.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
#使用pipeline将上下文传给模型
rag = pipeline(retriever_wrapper,chat)
WebModule(rag, history=[], port=8847).start().wait()
```
## 运行效果
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3256a34b21574285b6d07cba502f45ed.png)
效果还行 

后续大家可以加入语言对话来进一步丰富