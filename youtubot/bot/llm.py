# -*- coding: utf-8 -*-

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore import document
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import GigaChatEmbeddings
from langchain.chat_models.gigachat import GigaChat
from gigachat.exceptions import GigaChatException

from aiogram.exceptions import AiogramError

import os, re, asyncio, pathlib, logging, itertools, datetime
import gigachat
import youtube_transcript_api as yta
import config as cf


logger = logging.getLogger(__name__)

prompt_personalities = '''Ниже тебе предоставлен текст из видео:
{context}
Cоставь текст ответа на основе предоставленного текста из видео. Не говори что используешь предоставленный текст из видео для составления ответа!
НЕ ВЫСКАЗЫВАЙ МНЕ СВОЕГО ЛИЧНОГО МНЕНИЯ О ПРЕДОСТАВЛЕННОМ ТЕКСТЕ! НЕ ПРЕДЛАГАЙ МНЕ ВАРИАНТЫ ВОПРОСОВ.
Отвечай КРАТКО и максимально точно по представленному тексту из видео, не придумывай ничего от себя!'''
prompt_summarize = "Кратко в тезисах перескажи этот текст из видео, используя не более {max_words} слов"



class UsersLLMAnswer:
    """
    Используется для обработки пользовательских вопросов к youtube видео.
    Получает субтитры из youtube видео ролика и использует реализацию 
    RAG (Retrieval Augmented Generation) механизма для ответов на вопросы по видео

    Attributes:
        redis (Redis):                        Redis DB
        user_info (dict):                    Словарь контекстов пользователей
        embeddinger (Model):          Embeddings модель
        llm (Model):                          LLM чат модель

    Methods:
        llm_init()
            Инициализировать LLM Chat и Embeddings
        load_youtube_transcript(video_id) -> list, str
            Загрузить субтитры из youtube видео
        join_transcript_to_docs(transcript, max_chunk_size=500) -> list[YouTuDoc]
            Создать список документов из чанков субтитров
        async create_faiss(video_id) -> FAISS
            Создать или загрузить ранее созданое Faiss хранилище по id видео роликов
        get_video_id(text) -> str, list[str]
            Из всех youtube ссылок в тексте получить id видео роликов
        async summarize(msg, id, faiss)
            Поблочно сумаризовать и отправить пользователю текст из видео
        joiner(docs)
            Объеденить список чанков из векторного хранилища
        async create_chain(id) -> Сhain, FAISS
            Создать LLM цепочку для ответов на запросы
        async user_query(msg)
            Обработчик сообщений от всех пользователей
        async def a_query_answer(msg, chain, query)
            Передать `query' в LLM `chain` и результат отправить в бот
        async a_post(msg, text)
            Отправить текст в Telegram чат
        """
    
    def __init__(self, db):
        self.redis = db
        self.user_info = {}
        self.youtube_video_id = re.compile(r"(?:\S*youtu\.be\/(\S+)\?\S*)|(?:\S*youtube\.com\/watch\?v=(\S+))|(?:\S*youtube\.com\/live\/(\S+)\?\S*)")
        self.prompt_sum = ChatPromptTemplate.from_messages([
            ('system', prompt_summarize),
            ('human', '{text}'), ])
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', prompt_personalities),
            ('human', '{query}'), ])
        self.llm_init()


    def llm_init(self):
        """ Инициализировать LLM Chat и Embeddings"""
        self.embeddinger = GigaChatEmbeddings(
            scope="GIGACHAT_API_PERS",      # API для физ лиц (GIGACHAT_API_CORP для юр лиц)
            verify_ssl_certs=False )                 # без сертификата мин цифры (GIGACHAT_VERIFY_SSL_CERTS if verify_ssl_certs=True)
        self.llm = GigaChat(
            scope="GIGACHAT_API_PERS",      # API для физ лиц (GIGACHAT_API_CORP для юр лиц)
            verify_ssl_certs=False,                 # без сертификата мин цифры (GIGACHAT_VERIFY_SSL_CERTS if verify_ssl_certs=True)
            temperature=cf.temperature,
            streaming=True )                         # Потоковая передача


    def load_youtube_transcript(self, video_id):
        """ Загрузить субтитры из youtube видео
        
        Args:
            max_chunk_size (int): Максимальное кол-во символов для документов
        
        Returns:
            list: Список словарей с чанками субтитров и метаданными
            str: Код языка чанков субтитров
        """

        # Загрузить метаданные субтитров из youtube видео
        transcript_list = yta.YouTubeTranscriptApi.list_transcripts(video_id)
        generated = set()
        manual = set()

        for tra in transcript_list:
            if tra.is_generated:
                # Создать множество доступных языков текста субтитров, созданных youtube сервисом
                generated.add(tra.language_code)
            else:
                # Создать множество доступных языков текста субтитров, созданных автором
                manual.add(tra.language_code)
        if manual:
            language_code = 'en' if 'en' in manual else manual.pop()
        else:
            language_code = 'en' if 'en' in generated else generated.pop()
        
        # Загрузить субтитры из youtube видео
        transcript = yta.YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code]) # , proxies={"https": "https://user:pass@domain:port"} # format used by the requests library
        return transcript, language_code

    
    def join_transcript_to_docs(self, transcript, max_chunk_size=500):
        """ Создать список документов из чанков субтитров
        
        Args:
            transcript (list): Список словарей с чанками субтитров
            max_chunk_size (int): Максимальное кол-во символов для документов
                (default 500)
                
        Returns:
            list[YouTuDoc]: Список документов
        """
        transcripted = []
        for dc in transcript:
            dct = dc.copy()
            transcripted.append( YouTuDoc(page_content=dct.pop('text'), metadata=dct) )
        docs = [ YouTuDoc.zero_from(transcripted[0]) ]
        for doc in transcripted:
            if docs[-1].size() + doc.size() + 1 > max_chunk_size:
                if not docs[-1].size():
                    docs[-1], doc = doc.split(max_chunk_size)
                docs.append(doc)
            else:
                docs[-1] += doc
        return docs


    async def create_faiss(self, video_id):
        """ Создать или загрузить ранее созданое Faiss хранилище по id видео роликов
        
        Args:
            text (str): Текст пользовательского запроса
        
        Returns:
            FAISS: Векторное хранилище
        """
        faiss_pkl_file_path = pathlib.Path(cf.data_faiss_path) / f'{video_id}.faiss_pkl'
        faiss_pkl_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(faiss_pkl_file_path, 'rb') as fl:
                # Загрузить векторное хранилище из файла 
                search_index = FAISS.deserialize_from_bytes( embeddings=self.embeddinger, serialized=fl.read())
        except FileNotFoundError:
            
            # Загрузить субтитры из видео 
            transcript, language_code = self.load_youtube_transcript(video_id)

            # Создать список документов из чанков субтитров
            docs = self.join_transcript_to_docs(transcript, cf.max_chunk_size)

            # Создать векторное хранилище
            search_index = await FAISS.afrom_documents( docs, self.embeddinger )

            with open(faiss_pkl_file_path, 'wb') as fl:
                # Сохранить в файл векторное хранилище
                fl.write(search_index.serialize_to_bytes())    # serializes the faiss
        return search_index


    def get_video_id(self, text):
        """ Из всех youtube ссылок в тексте получить id видео роликов
        
        Args:
            text (str): Текст пользовательского запроса
        
        Returns:
            str: Текст без youtube ссылок
            list[str]: Cписок id видео роликов
        """
        ids = []
        spans = [0]
        for location in self.youtube_video_id.finditer(text):
            for v in location.groups():
                if v is not None:
                    ids.append(v)
                    break
            spans += location.span()
        if ids:
            spans.append(len(text))
            it = iter(spans)
            text = ''.join(text[a:b] for a,b in zip(it,it)).strip()
        return text, ids


    async def summarize(self, msg, id, faiss):
        """ Поблочно сумаризовать и отправить пользователю Faiss хранилище,
                созданное из видео с `id`
        
        Args:
            msg (Message): Aiogram сообщение пользователя
            id (str): Id виде из youtube
        
        Returns:
            None
        """
        redis_key = f'youtube/{id}'
        
        # Получить сохраненную сумаризацию из БД
        text = await self.redis.get(redis_key)
        if text:
            for chunk in text.split('\n'):
                await self.a_post(msg, chunk)
            return

        # Получить список, сортированых по времени, чанков текста субтитров из Faiss хранилища
        docs = list(sorted(faiss.docstore._dict.values(), key=lambda x: x.metadata['start']))
        join_docs = []

        # Сделать каждый блок для сумаризации из `cf.chunks_by_query_from_faiss` штук Faiss чанков 
        for i in range(0, len(docs), cf.chunks_by_query_from_faiss):
            chunks = '\n'.join(d.page_content for d in docs[i:i+cf.chunks_by_query_from_faiss])
            join_docs.append( chunks )

        # LLM цепочка для сумаризации
        chain = self.prompt_sum | self.llm | StrOutputParser()  
        chunks = []
        for doc in join_docs:
            # Сумаризовать блок текста до 25 слов приблизительно
            text = await self.a_query_answer(msg, chain, {'max_words': '25', 'text': doc})
            if text:
                chunks.append(text)
        if chunks:
            # Объеденить сумаризованные блоки и отправить в БД
            await self.redis.set(redis_key, '\n'.join(chunks))


    def joiner(self, docs):
        """ Объеденить список чанков из векторного хранилища """
        sorted_docs = list(sorted(docs, key=lambda x: x.metadata['start']))
        return ' '.join(d.page_content for d in sorted_docs)


    async def create_chain(self, id):
        """ Создать LLM цепочку для ответов на запросы

        Args:
            id (str): Id виде из youtube
        
        Returns:
            Сhain: LLM цепочка
            FAISS: Векторное хранилище
        """
        faiss = await self.create_faiss(id)
        retrieval_pass = RunnableParallel( {
            "context": faiss.as_retriever(k=cf.chunks_by_query_from_faiss) | self.joiner,     # as_retriever(search_kwargs={'k': 3})
            "query": RunnablePassthrough() })
        chain = retrieval_pass | self.prompt | self.llm | StrOutputParser()
        #~ chain = self.llm | StrOutputParser()
        return chain, faiss


    async def user_query(self, msg):
        """ Обработчик сообщений от всех пользователей

        Args:
            msg (Message): Aiogram сообщение пользователя
        
        Returns:
            None
        """

        # получить ссылку (если она есть) на ютуб видео из запроса
        query, ids = self.get_video_id(msg.text)
        #~ print(query, ids)
        if ids:

            # если запрос содержит ссылки на видео
            if len(ids) > 1:
                await self.a_post(msg, 'Я запомнил видео по последней ссылке, потому что '
                                                        'могу просмотреть только одно видео за раз.')
            else:
                await self.a_post(msg, 'Я запомнил это видео.')

            # получить контекст пользователя по id
            user = self.user_info.setdefault(msg.from_user.id, {})

            # получить ссылку на faiss db и объект langchain
            user['chain'], faiss = await self.create_chain(ids[-1])
            if not query:
                # если в запросе была только ссылка
                # сумаризовать весь ролик
                await self.summarize(msg, ids[-1], faiss)
                await self.a_post(msg, 'Я могу рассказать еще что-нибудь по этому видео.')
                return

        if (user := self.user_info.get(msg.from_user.id)) is None:
            # если контекст пользователя еще не создан
            await self.a_post(msg, 'Я не знаю к какому видео относится запрос.\n'
                                                    'Пришлите пожалуйста ссылку на видео из YouTube.')
        else:
            await self.a_query_answer(msg, user['chain'], query)


    async def a_query_answer(self, msg, chain, query):
        """ Передать `query' в LLM `chain` и результат отправить в бот

        Args:
            msg (Message): Aiogram сообщение пользователя
            chain (Сhain): LLM цепочка
            query (Any): Объект запроса к LLM
       
        Returns:
            str: Ответ LLM
        """
        try:
            # получить ответ от LLM на запрос полльзователя
            text = await chain.ainvoke(query)
        except GigaChatException:
            # Запись в лог всех ошибок aiogram при отправке в чат
            logger.exception('GigaChatException!')
        else:
            await self.a_post(msg, text)
            return text


    async def a_post(self, msg, text):
        """ Отправить текст в Telegram чат """
        try:
            await msg.answer(text)
        except AiogramError:
            # Запись в лог всех ошибок aiogram при отправке в чат
            logger.exception('AiogramError!')
    

if os.getenv('HUGGING_FACE_EMBEDDINGS'):
    
    class UsersLLMAnswer(UsersLLMAnswer):

        def llm_init(self):
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddinger = HuggingFaceEmbeddings(
                model_name=os.environ['HUGGING_FACE_EMBEDDINGS'],
                model_kwargs={"device": 'cpu'},	
                encode_kwargs={"normalize_embeddings": True})
            self.llm = GigaChat(
                scope="GIGACHAT_API_PERS",      # API для физ лиц (GIGACHAT_API_CORP для юр лиц)
                verify_ssl_certs=False,                 # без сертификата мин цифры (GIGACHAT_VERIFY_SSL_CERTS if verify_ssl_certs=True)
                temperature=cf.temperature,
                streaming=True )                         # Потоковая передача



class YouTuDoc(document.Document):
    """
    The class is designed for convenient gluing and separating segments of subtitles for YouTube videos.
    """
    
    def split(self, max_size):
        """ Разделить `page_content` по границам слов на две части.
        Вернуть 2 документа, где первая часть `page_content` не более `max_size` символов
        """
        separator = '.'
        chunks = self.page_content.split(separator)
        size = 0
        for i,st in enumerate(chunks):
            if size + len(st) + 1 > max_size:
                break
            size += len(st) + 1
        first_str = chunks[:i]
        second_str = chunks[i:]
        if second_str:
            first_str.append('')
        first_str = separator.join(first_str)
        second_str = separator.join(second_str)
        first_metadata = self.metadata.copy()
        first_metadata['duration'] *= len(first_str) / self.size() 
        second_metadata = self.metadata.copy()
        second_metadata['start'] += first_metadata['duration']
        second_metadata['duration'] -= first_metadata['duration']
        first = self.__class__(page_content=first_str, metadata=first_metadata)
        second = self.__class__(page_content=second_str, metadata=second_metadata)
        return first, second


    def __add__(self, other):
        """ Объеденить `page_content` двух документов."""
        metadata = self.metadata.copy()
        metadata['duration'] += other.metadata['duration']
        res = self.__class__(page_content=' '.join((self.page_content,other.page_content)), metadata=metadata)
        return res


    @classmethod
    def copy_from(Cls, other):
        """ Создать копию `other` документа."""
        return Cls(page_content=other.page_content, metadata=other.metadata.copy())


    @classmethod
    def zero_from(Cls, other):
        """ Создать пустой документ из `other`, с нулевой `metadata`."""
        return Cls(page_content='', metadata=dict.fromkeys(other.metadata.keys(), 0))


    def size(self):
        """ Количество символов в `page_content`."""
        return len(self.page_content)


