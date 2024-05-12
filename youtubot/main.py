# -*- coding: utf-8 -*-
#! /usr/bin/python3

# pip install redis aiogram python-dotenv gigachat faiss-cpu langchain langchain-community youtube_transcript_api
# pip install sentence-transformers     # for HuggingFaceEmbeddings

import dotenv
dotenv.load_dotenv()

import os, asyncio, logging
import redis.asyncio as redis
import aiogram as ag
import config as cf
from bot.handlers import users

logger = logging.getLogger(__name__)


async def main():
    if os.getenv('DEBUG'):
        logging.basicConfig( level=logging.DEBUG, format=cf.logging_format )

    # Connect to redis
    db = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=os.environ['REDIS_PORT'], 
        password=os.getenv('REDIS_PASSWORD'),
        decode_responses=True)                                                                                      # automatically convert responses from bytes to strings
    logger.info('Redis connected.')

    try:
        bot = ag.Bot(token=os.environ['TELEGRAM_BOT_TOKEN'])
        dp = ag.Dispatcher(bot=bot, storage=ag.fsm.storage.memory.MemoryStorage())
        await users.setup(bot, dp, db)
        logger.info('users.setup is done.')

        await bot.delete_webhook(drop_pending_updates=True)                                    # delete all messages sent while the bot was not working

        logger.info('Starting polling...')
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await db.aclose()


if __name__ == '__main__':
    asyncio.run(main())






