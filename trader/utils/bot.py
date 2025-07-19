import logging
from threading import Event
from telegram.ext import Updater, MessageHandler, Filters

from ..config import NotifyConfig
from .objects.data import TradeData
from .database import db
from .database.tables import SecurityInfo

stop_flag = Event()
pause_flag = Event()


class TelegramBot:
    def __init__(self, account_name: str):
        self.account_name = account_name
        self.chat_id = NotifyConfig.TELEGRAM_CHAT_ID
        self.updater = Updater(
            token=NotifyConfig.TELEGRAM_TOKEN, use_context=True)

        dispatcher = self.updater.dispatcher
        dispatcher.add_handler(
            MessageHandler(Filters.text & ~Filters.command, self.handle_msg))
        self.updater.start_polling()

    def post(self, context, msg: str):
        context.bot.send_message(
            chat_id=NotifyConfig.TELEGRAM_CHAT_ID,
            text=msg
        )

    def handle_msg(self, update, context):
        msg = update.message.text.strip()

        logging.warning(f'[Message Received] {msg}')
        if self.account_name not in msg:
            return

        if "暫停交易" in msg or "暫停監控" in msg:
            pause_flag.set()
            self.post(context, msg="🛑 已暫停監控")

        elif "繼續交易" in msg or "繼續監控" in msg:
            pause_flag.clear()
            self.post(context, msg="✅ 已恢復監控")

        elif "停止交易" in msg or "停止監控" in msg:
            stop_flag.set()
            self.post(context, msg='❌ 程式即將停止')

        elif "監控狀態" in msg:
            if stop_flag.is_set():
                status = "❌ 已關閉"
            elif pause_flag.is_set():
                status = "🛑 暫停交易中"
            else:
                status = "✅ 交易中"
            self.post(context, msg=f"目前狀態：{status}")

        elif "目前部位" in msg or "當前部位" in msg:
            position = db.query(
                SecurityInfo,
                SecurityInfo.mode == TradeData.Account.Mode,
                SecurityInfo.account == self.account_name
            )[['code', 'quantity']]
            position = position.groupby('code').quantity.sum().to_dict()

            self.post(context, msg=f'{position}')
