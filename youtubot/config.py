# -*- coding: utf-8 -*-

# формат вывода логера
logging_format                      = f"%(asctime)s [%(levelname)s:%(name)s] - %(message)s -> %(pathname)s, line %(lineno)d, in %(funcName)s"

# файлы векторного хранилища
data_faiss_path                     = 'data/faiss_db'

# кол-во символов в одном чанке векторного хранилища
max_chunk_size                    = 500

# кол-во чанков ближайших к запросу, выбираемых из векторного хранилища
chunks_by_query_from_faiss = 6

# температурный коэфициент LLM модели
temperature                          = 10**-3

try:
    import security_keys
except ImportError:
    pass