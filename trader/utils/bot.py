import logging
from threading import Event, Thread
from typing import Optional

from telegram import Update
from telegram.error import NetworkError, TimedOut, RetryAfter, Conflict, TelegramError
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from ..config import NotifyConfig, StrategyList
from .notify import Notification
from .objects.data import TradeData
from .database import db
from .database.tables import SecurityInfo

# Status flag: supports multiple accounts
stop_flags = {}
pause_flags = {}

# chat_id whitelist
WHITELIST = {str(x) for x in NotifyConfig.TELEGRAM_CHAT_ID.values()}


class TelegramBot:
    def __init__(self, account_name: str):
        self.account_names = [account_name]
        self._init_flags()

        # dict 判斷不要用 "is {}"
        token = None
        try:
            token = NotifyConfig.TELEGRAM_TOKEN.get(account_name, "")
        except Exception:
            token = ""

        if not token:
            logging.warning("Telegram token 未設定或為空，跳過啟動 bot。")
            self.app = None
            return

        try:
            for name in (
                "httpx",            # HTTP 請求「HTTP Request: POST ...」
                "httpcore",         # httpx 的底層連線
                "telegram",         # PTB 自身（含 network/request）
                "telegram.ext",
                "apscheduler",      # 若也不想看到 APScheduler 訊息
            ):
                logging.getLogger(name).setLevel(logging.WARNING)

            self.app = (
                ApplicationBuilder()
                .token(token)
                .connect_timeout(300)
                .read_timeout(300)
                .write_timeout(30)
                .build()
            )

            # 指令
            self.app.add_handler(CommandHandler("pause", self.cmd_pause))
            self.app.add_handler(CommandHandler("resume", self.cmd_resume))
            self.app.add_handler(CommandHandler("stop", self.cmd_stop))
            self.app.add_handler(CommandHandler("status", self.cmd_status))
            self.app.add_handler(CommandHandler("position", self.cmd_position))
            self.app.add_handler(CommandHandler(
                "update_max_qty", self.cmd_update_max_qty))
            self.app.add_handler(MessageHandler(
                filters.Regex(r"^/update-max-qty(?:\s|$)"),
                self.cmd_update_max_qty)
            )

            # 後備文字處理器（防漏掉）
            self.app.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND, self.handle_text)
            )

            # 錯誤處理
            self.app.add_error_handler(self.error_handler)

            # 非阻塞背景啟動（主程式可繼續跑）
            self._thread = Thread(
                target=self._run_polling_in_background, daemon=True
            )
            self._thread.start()

        except Conflict as e:
            logging.error(f"Telegram Conflict: {e}")
            self.app = None
        except Exception:
            logging.exception("Initialize telegram application failed:")
            self.app = None

    def _run_polling_in_background(self):
        """
        以背景 thread 執行 run_polling()，避免阻塞主執行緒。
        停止訊號交由主程式結束時一併處理（daemon thread）。
        """
        try:
            # stop_signals=None 可避免攔截主程式訊號；allowed_updates 可留空用預設
            self.app.run_polling(
                stop_signals=None,
                poll_interval=1.0,              # 失敗後重試間隔
                allowed_updates=None,           # 預設即可
                drop_pending_updates=False      # 依需求；若常重啟可設 True
            )
        except Exception:
            logging.exception("run_polling 發生例外：")

    def _init_flags(self):
        for name in self.account_names:
            stop_flags[name] = Event()
            pause_flags[name] = Event()

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        err = context.error

        # 這些都屬於暫時性或可預期：降噪
        transient = (NetworkError, TimedOut, RetryAfter, Conflict)
        if isinstance(err, transient):
            logging.warning("Telegram 暫時性網路異常：%s", err)
            return

        if isinstance(err, TelegramError) and "terminated by other getUpdates request" in str(err):
            logging.warning(
                "⚠️ Bot polling 被其他實例中斷（terminated by other getUpdates request）")
            # 這裡可視需要做應對；run_polling 會自動處理重連
            return

        logging.exception("🚨 Handler 未預期錯誤", exc_info=err)

    async def post(self, update: Update, msg: str):
        if update and update.effective_message:
            await update.effective_message.reply_text(text=msg)

    def _check_permission(self, update: Optional[Update]) -> bool:
        """回傳是否允許；白名單為空時一律放行；不允許時回覆 chat_id 方便加入白名單。"""
        try:
            if not update or not update.effective_chat:
                return False

            chat_id = str(update.effective_chat.id)
            msg = (update.effective_message.text or "").strip(
            ) if update.effective_message else ""
            logging.info(f"[TEXT MESSAGE][{chat_id}] {msg}")

            # 白名單為空 -> 放行（避免因設定缺漏導致完全不回覆）
            if not WHITELIST:
                logging.warning("⚠️ Telegram 白名單為空，允許所有聊天。")
                return True

            # 在白名單 -> 放行
            if chat_id in WHITELIST:
                return True

            # 允許以 username 白名單（可選，用於群組/超級群）
            try:
                user = update.effective_user
                uname = (user.username or "").lower() if user else ""
                user_whitelist = {u.lower() for u in getattr(
                    NotifyConfig, "TELEGRAM_USER_WHITELIST", [])}
                if uname and uname in user_whitelist:
                    return True
            except Exception:
                pass

            # 不允許 -> 告知 chat_id 以便加入白名單
            if update.effective_message:
                update.effective_message.reply_text(
                    f"⛔️ 未授權聊天（chat_id={chat_id}）。"
                )
            return False
        except Conflict:
            return False

    def _parse_args(self, context: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
        try:
            # v20+ 用 context.args 取參數
            if context.args:
                return context.args[0]

            if self.account_names:
                return self.account_names[0]

            return None
        except Exception:
            return None

    def _parse_update_max_qty_args(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if context.args:
            raw = ' '.join(context.args)
        elif update and update.effective_message:
            raw = (update.effective_message.text or '').strip()
            raw = raw.split(maxsplit=1)[1] if ' ' in raw else ''
        else:
            raw = ''

        parts = raw.rsplit('-', 2)
        if len(parts) != 3 or '-' not in parts[0]:
            return None

        account_strategy, target, max_qty = parts
        account, strategy = account_strategy.split('-', 1)
        return account.strip(), strategy.strip(), target.strip(), max_qty.strip()

    # ========== Command Response (async) ==========

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_permission(update):
            return

        name = self._parse_args(context)
        if name and name in pause_flags:
            pause_flags[name].set()
            await self.post(update, f"🛑 [{name}] 已暫停監控")
        else:
            await self.post(update, "⚠️ 請輸入帳號名稱，例如 `/pause account_name`")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_permission(update):
            return

        name = self._parse_args(context)
        if name and name in pause_flags:
            pause_flags[name].clear()
            await self.post(update, f"✅ [{name}] 已恢復監控")
        else:
            await self.post(update, "⚠️ 請輸入帳號名稱，例如 `/resume account_name`")

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_permission(update):
            return

        name = self._parse_args(context)
        if name and name in stop_flags:
            stop_flags[name].set()
            await self.post(update, f"❌ [{name}] 程式即將停止")
        else:
            await self.post(update, "⚠️ 請輸入帳號名稱，例如 `/stop account_name`")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_permission(update):
            return

        name = self._parse_args(context)
        if name in stop_flags and name in pause_flags:
            if stop_flags[name].is_set():
                status = "❌ 已關閉"
            elif pause_flags[name].is_set():
                status = "🛑 暫停中"
            else:
                status = "✅ 交易中"
            await self.post(update, f"[{name}] 目前狀態：{status}")
        else:
            await self.post(update, "⚠️ 請輸入帳號名稱，例如 `/status account_name`")

    async def cmd_position(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_permission(update):
            return

        name = self._parse_args(context)
        if name:
            # 注意：這是同步查詢，若耗時可考慮丟到 ThreadPool
            position = db.query(
                SecurityInfo,
                SecurityInfo.mode == TradeData.Account.Mode,
                SecurityInfo.account == name
            )[['code', 'quantity']]
            pos_dict = position.groupby('code').quantity.sum().to_dict()
            await self.post(update, msg=f"📦 [{name}] 持倉：{pos_dict}")
        else:
            await self.post(update, "⚠️ 請輸入帳號名稱，例如 `/position account_name`")

    async def cmd_update_max_qty(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_permission(update):
            return

        parsed = self._parse_update_max_qty_args(update, context)
        usage = "Usage: /update_max_qty account-strategy-target-max_qty"
        if parsed is None:
            await self.post(update, usage)
            return

        account, strategy, target, max_qty = parsed
        if account not in self.account_names:
            await self.post(update, f"Unknown account: {account}")
            return

        try:
            max_qty = int(max_qty)
        except ValueError:
            await self.post(update, f"Invalid max_qty: {max_qty}")
            return

        if max_qty < 0:
            await self.post(update, "max_qty must be >= 0")
            return

        conf = StrategyList.Config.get(strategy)
        if conf is None:
            await self.post(update, f"Unknown strategy: {strategy}")
            return

        if not hasattr(conf, 'max_qty'):
            await self.post(update, f"Strategy has no max_qty: {strategy}")
            return

        old_qty = conf.max_qty.get(target)
        conf.max_qty[target] = max_qty
        logging.info(
            f'[Strategy max_qty]Update|{account}|{strategy}|{target}|{old_qty}->{max_qty}')
        notifier = Notification(NotifyConfig, account=account)
        notifier.post_update_max_qty(target, max_qty)
        await self.post(update, f"【更新部位】最大數量\n{target}: {max_qty}")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._check_permission(update):
            return

        await self.post(update, "❓ 請使用正確指令，如：`/pause account_name`")

    def get_flags(self, account_name: str):
        '''========== Get status flags of an account =========='''
        return stop_flags[account_name], pause_flags[account_name]
