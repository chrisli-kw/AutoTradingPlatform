# TelegramBot 使用說明

本模組提供透過 Telegram Bot 遠端控制交易監控程式的能力，支援暫停、恢復、停止、狀態查詢、庫存查詢，以及盤中查詢/更新策略 `max_qty`。

目前實作位於 `trader/utils/bot.py`，使用 `python-telegram-bot` v20+ async API，並以背景 thread 啟動 polling，因此不會阻塞交易主迴圈。

---

## 功能概覽

- 支援單一監控帳號的 pause/resume/stop/status 控制
- 支援查詢帳號持倉彙總
- 支援盤中查詢策略商品最大口數 `max_qty`
- 支援盤中更新策略商品最大口數 `max_qty`
- 使用 `NotifyConfig.TELEGRAM_CHAT_ID` 建立 chat_id 白名單
- 白名單為空時會放行所有聊天，並在 log 中警告
- 使用 `stop_flags` / `pause_flags` 與主監控迴圈同步控制狀態
- 內建 Telegram 網路暫時性錯誤降噪

---

## 設定方式

在 `lib/config.ini` 的 `[NOTIFY]` 區塊設定 Telegram Token 與允許操作的 chat_id：

```ini
[NOTIFY]
PLATFORM = Telegram
TELEGRAM_TOKEN = {"account_name": "your_bot_token"}
TELEGRAM_CHAT_ID = {"account_name": "your_chat_id"}
LINE_TOKEN = your_LINE_Notify_token
```

`TELEGRAM_TOKEN` 和 `TELEGRAM_CHAT_ID` 都是以帳號名稱為 key 的 JSON 字串。`TelegramBot(account_name)` 會使用對應帳號的 token 啟動 bot。

目前 `requirements.txt` 使用：

```txt
python-telegram-bot==22.5
```

---

## 初始化方式

交易監控程式中初始化：

```python
from trader.utils.bot import TelegramBot

bot = TelegramBot(account_name="account_name")
stop_flag, pause_flag = bot.get_flags("account_name")
```

主迴圈可用 flags 控制：

```python
while not stop_flag.is_set():
    if pause_flag.is_set():
        time.sleep(1)
        continue

    execute_strategy()
```

如果 token 未設定或為空，bot 會跳過啟動並寫入 warning log，不會中斷主程式。

---

## 指令總覽

所有指令都需要由白名單 chat_id 發送。

| 指令                                                      | 功能                 | 範例                                                |
| --------------------------------------------------------- | -------------------- | --------------------------------------------------- |
| `/pause <account>`                                        | 暫停該帳號監控       | `/pause account`                                    |
| `/resume <account>`                                       | 恢復該帳號監控       | `/resume account`                                   |
| `/stop <account>`                                         | 停止該帳號監控       | `/stop account`                                     |
| `/status <account>`                                       | 查詢監控狀態         | `/status account`                                   |
| `/position <account>`                                     | 查詢持倉彙總         | `/position account`                                 |
| `/check_max_qty <account>-<strategy>-<target>`            | 查詢策略商品最大口數 | `/check_max_qty account-LongStrategy-TMF202507`     |
| `/update_max_qty <account>-<strategy>-<target>-<max_qty>` | 更新策略商品最大口數 | `/update_max_qty account-LongStrategy-TMF202507-10` |

`/update-max-qty ...` 也會被導到更新 `max_qty` 的處理流程。

---

## 監控控制指令

### 暫停監控

```text
/pause account
```

回覆：

```text
🛑 [<name>] 已暫停監控
```

### 恢復監控

```text
/resume account
```

回覆：

```text
✅ [<name>] 已恢復監控
```

### 停止監控

```text
/stop account
```

回覆：

```text
❌ [<name>] 程式即將停止
```

### 查詢狀態

```text
/status account
```

可能回覆：

```text
[<name>] 目前狀態：✅ 交易中
[<name>] 目前狀態：🛑 暫停中
[<name>] 目前狀態：❌ 已關閉
```

---

## 持倉查詢

```text
/position account
```

Bot 會從 `SecurityInfo` 查詢指定帳號在目前 `TradeData.Account.Mode` 下的庫存，並依商品代碼彙總：

```text
📦 [<name>] 持倉：{'TMF202507': 3, 'TXF202507': 1}
```

---

## max_qty 查詢與更新

### 查詢 max_qty

```text
/check_max_qty account-LongStrategy-TMF202507
```

回覆：

```text
【chrisli_1 確認部位】最大數量
TMF202507: 10
```

### 更新 max_qty

```text
/update_max_qty account-LongStrategy-TMF202507-12
```

回覆：

```text
【chrisli_1 更新部位】最大數量
TMF202507: 12
```

更新流程會：

1. 確認 account 是否為目前 bot 綁定帳號
2. 從 `StrategyList.Config` 取得策略設定物件
3. 檢查策略是否有 `max_qty`
4. 更新記憶體中的 `conf.max_qty[target]`
5. 呼叫策略的 `update_max_qty(account, strategy, target, max_qty)` 持久化更新

策略若要支援盤中持久化更新，需提供：

```python
max_qty = {}

@staticmethod
def update_max_qty(account_name: str, strategy: str, target: str, max_qty: int):
    ...
```

---

## 權限與安全性

白名單來源：

```python
WHITELIST = {str(x) for x in NotifyConfig.TELEGRAM_CHAT_ID.values()}
```

行為：

- chat_id 在白名單內才允許操作
- 若白名單為空，bot 會放行所有聊天並記錄 warning
- 未授權聊天會收到包含 chat_id 的訊息，方便加入設定
- 也可透過 `NotifyConfig.TELEGRAM_USER_WHITELIST` 支援 username 白名單

---

## 常見錯誤回覆

參數格式錯誤：

```text
Usage: /update_max_qty account-strategy-target-max_qty
Usage: /check_max_qty account-strategy-target
```

帳號不符合目前 bot：

```text
Unknown account: account_name
```

策略不存在：

```text
Unknown strategy: strategy_name
```

策略不支援 `max_qty`：

```text
Strategy has no max_qty: strategy_name
```

`max_qty` 不是整數或小於 0：

```text
Invalid max_qty: abc
max_qty must be >= 0
```

未知文字訊息：

```text
❓ 請使用正確指令，如：`/pause account_name`
```

---

## 注意事項

- 同一個 Telegram bot token 不應同時由多個程式 polling，否則 Telegram 可能回傳 `terminated by other getUpdates request`。
- `TelegramBot` 以 daemon thread 啟動，主程式結束時 polling thread 會一起結束。
- `/position` 查詢依賴資料庫 `SecurityInfo` 與目前 `TradeData.Account.Mode`。
- `/update_max_qty` 適合盤中調整策略風控，但策略本身仍應在下單前檢查目前持倉與可下單口數。
