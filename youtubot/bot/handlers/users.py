# -*- coding: utf-8 -*-

import asyncio, aiogram as ag
from typing import Any, Callable, Dict, Awaitable
from aiogram.types import Message, TelegramObject
from aiogram.utils.chat_action import ChatActionSender
from .. import llm

router = ag.Router()
router.message.filter( ag.F.chat.type == 'private' )    	            # filter for private chats only


@router.message(ag.F.text, ag.filters.Command("start"))
async def start_handler(msg: Message):                              # handler of the very first command '/start'
    await msg.answer(   "Здравствуйте! Я могу посмотреть любой ролик из YouTube и рассказать вам о нем. " \
                                    "Пришлите мне ссылку на ролик и задавайте любые вопросы.")


@router.message(ag.F.text)#, flags={'chat_action': 'typing'})
async def message_handler(msg: Message, llm_answer):    # handler for all other private messages
    # Release the aiogram task and pass the user query in a separate task
    await llm_answer.user_query(msg)


class UsersMiddleware(ag.BaseMiddleware):
    
    def __init__(self, llm_answer):
        super().__init__()
        self.llm_answer = llm_answer
    
    async def __call__(
        self,
        handler: Callable[[ag.types.TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: ag.types.TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        # Pass a link to the llm task to each user request
        data['llm_answer'] = self.llm_answer
        # Create status "printing..."
        async with ChatActionSender( bot=data['bot'], chat_id=event.chat.id, action='typing'):
            # Call the appropriate message handler
            return await handler(event, data)		

async def setup(bot, dispatcher, db):
    router.message.middleware( UsersMiddleware(llm.UsersLLMAnswer(db)) )
    dispatcher.include_routers(router)
