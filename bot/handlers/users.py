# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, Awaitable
import asyncio, logging

import aiogram as ag
from aiogram.utils.chat_action import ChatActionMiddleware

logger = logging.getLogger(__name__)

router = ag.Router()
router.message.middleware( ChatActionMiddleware() )
#~ router.message.filter( ag.F.chat.type.is_('private') )	# for private chats only


class UsersMarkMiddleware(ag.BaseMiddleware):
	
	def __init__(self):
		super().__init__()
		self.users = {}

	async def __call__(
		self,
		handler: Callable[[ag.types.TelegramObject, Dict[str, Any]], Awaitable[Any]],
		event: ag.types.TelegramObject,
		data: Dict[str, Any],
	) -> Any:
		data["user_info"] = self.users.setdefault(data["event_from_user"].id, {})
		return await handler(event, data)

@router.message(flags={'chat_action': 'typing'})
async def message_handler(msg: ag.types.Message, user_info: dict):
	user_info['text'] = user_info.setdefault('text', '') + msg.text
	logger.info(user_info['text'])
	await msg.answer(msg.text)


async def setup(bot, dispatcher):
	dispatcher.update.outer_middleware( UsersMarkMiddleware() )
	dispatcher.include_routers(router)
