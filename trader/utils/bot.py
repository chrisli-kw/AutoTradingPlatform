import logging
from threading import Event
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from ..config import NotifyConfig
from .objects.data import TradeData
from .database import db
from .database.tables import SecurityInfo

# Status flag: supports multiple accounts
stop_flags = {}
pause_flags = {}

# chat_id whitelist
WHITELIST = {NotifyConfig.TELEGRAM_CHAT_ID}


class TelegramBot:
    def __init__(self, account_names: list):
        self.account_names = account_names
        self.updater = Updater(
            token=NotifyConfig.TELEGRAM_TOKEN, use_context=True)
        dispatcher = self.updater.dispatcher

        # Bot commant dispatcher
        dispatcher.add_handler(CommandHandler("pause", self.cmd_pause))
        dispatcher.add_handler(CommandHandler("resume", self.cmd_resume))
        dispatcher.add_handler(CommandHandler("stop", self.cmd_stop))
        dispatcher.add_handler(CommandHandler("status", self.cmd_status))
        dispatcher.add_handler(CommandHandler("position", self.cmd_position))

        # 後備文字處理器（防漏掉）
        dispatcher.add_handler(MessageHandler(
            Filters.text & ~Filters.command, self.handle_text))

        self.updater.start_polling()
        self._init_flags()

    def _init_flags(self):
        for name in self.account_names:
            stop_flags[name] = Event()
            pause_flags[name] = Event()

    def post(self, update, msg: str):
        update.message.reply_text(text=msg)

    def _check_permission(self, update):
        if update.message is None:
            return False

        chat_id = str(update.effective_chat.id)
        msg = update.message.text.strip()
        logging.warning(f'[TEXT MESSAGE][{chat_id}] {msg}')
        return chat_id in WHITELIST

    def _parse_args(self, update):
        try:
            return update.message.text.strip().split()[1]
        except IndexError:
            self.post(update, msg="⚠️ 請輸入帳號名稱，例如 `/pause account_name`")
            return None

    # ========== Command Response ==========
    def cmd_pause(self, update, context):
        if not self._check_permission(update):
            return

        name = self._parse_args(update)
        if name and name in pause_flags:
            pause_flags[name].set()
            self.post(update, f"🛑 [{name}] 已暫停監控")

    def cmd_resume(self, update, context):
        if not self._check_permission(update):
            return

        name = self._parse_args(update)
        if name and name in pause_flags:
            pause_flags[name].clear()
            self.post(update, f"✅ [{name}] 已恢復監控")

    def cmd_stop(self, update, context):
        if not self._check_permission(update):
            return

        name = self._parse_args(update)
        if name and name in stop_flags:
            stop_flags[name].set()
            self.post(update, f"❌ [{name}] 程式即將停止")

    def cmd_status(self, update, context):
        if not self._check_permission(update):
            return

        name = self._parse_args(update)
        if name in stop_flags and name in pause_flags:
            if stop_flags[name].is_set():
                status = "❌ 已關閉"
            elif pause_flags[name].is_set():
                status = "🛑 暫停中"
            else:
                status = "✅ 交易中"
            self.post(update, f"[{name}] 目前狀態：{status}")

    def cmd_position(self, update, context):
        if not self._check_permission(update):
            return

        name = self._parse_args(update)
        if name:
            position = db.query(
                SecurityInfo,
                SecurityInfo.mode == TradeData.Account.Mode,
                SecurityInfo.account == name
            )[['code', 'quantity']]
            pos_dict = position.groupby('code').quantity.sum().to_dict()
            self.post(update, msg=f"📦 [{name}] 持倉：{pos_dict}")

    def handle_text(self, update, context):
        '''========== Handle text messages =========='''

        if not self._check_permission(update):
            return

        self.post(update, msg="❓ 請使用正確指令，如：`/pause account_name`")

    def get_flags(self, account_name: str):
        '''========== Get status flags of an account =========='''
        return stop_flags[account_name], pause_flags[account_name]
