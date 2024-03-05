# -*- coding: utf-8 -*-
#! /usr/bin/python3


import os, asyncio, logging
import aiogram as ag
import config as cf
from bot.handlers import users

logger = logging.getLogger(__name__)

async def main():
	bot = ag.Bot(token=os.environ['TELEGRAM_BOT_TOKEN'])
	dp = ag.Dispatcher(bot=bot, storage=ag.fsm.storage.memory.MemoryStorage())

	logger.info("Configure handlers...")
	await users.setup(bot, dp)

	await bot.delete_webhook(drop_pending_updates=True)		# delete all messages sent while the bot was not working
	await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

if __name__ == '__main__':
	asyncio.run(main())






