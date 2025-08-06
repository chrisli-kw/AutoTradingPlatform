# 📡 TelegramBot 使用說明

本模組提供透過 Telegram 機器人遠端控制交易系統的能力，支援多帳號、交易狀態控制、持倉查詢，並內建 chat_id 白名單驗證。

---

## ✅ 功能概覽

- ✅ 支援多帳號控制（例如 `account1`, `account2`）
- ✅ 支援 `/pause`, `/resume`, `/stop`, `/status`, `/position` 指令格式
- ✅ Telegram `chat_id` 白名單驗證機制，提升安全性
- ✅ 內建共享旗標 `stop_flag`, `pause_flag` 控制策略流程
- ✅ 可擴充更多帳號或策略控制項目

---

## 🔧 初始化方式

在主程式中初始化 Telegram 機器人並註冊帳號：

```python
from telegram_bot import TelegramBot

bot = TelegramBot(account_names=["account1", "account2"])

# 取得旗標（用於主迴圈中控制該帳號是否執行）
stop_flag, pause_flag = bot.get_flags("account")
```

> `get_flags(account_name)` 會傳回該帳號對應的 `Event()` 控制物件。

---

## 📬 指令格式與行為說明

以下指令需透過 Telegram 機器人傳送，且必須由「白名單」內的 `chat_id` 使用者傳送：

| 指令格式           | 功能描述               | 範例                |
| ------------------ | ---------------------- | ------------------- |
| `/pause <帳號>`    | 暫停該帳號的策略執行   | `/pause account`    |
| `/resume <帳號>`   | 恢復該帳號的策略執行   | `/resume account`   |
| `/stop <帳號>`     | 關閉該帳號整個監控程式 | `/stop account`     |
| `/status <帳號>`   | 查詢該帳號目前執行狀態 | `/status account`   |
| `/position <帳號>` | 查詢該帳號目前持倉資訊 | `/position account` |

---

## 📊 狀態說明

當執行 `/status <帳號>` 指令時，系統會回覆以下其中一種狀態：

- `✅ 交易中`：策略正在正常執行中
- `🛑 暫停中`：策略已被暫停，但程式尚未退出
- `❌ 已關閉`：策略程式已被關閉（flag 已 set）

---

## 🔐 安全性機制

- 僅允許 `WHITELIST` 中的 chat_id 發送指令（定義於程式中 `WHITELIST` 集合）
- 機器人不處理來源未知的訊息（即使格式正確）

---

## 🧪 主程式範例整合

```python
bot = TelegramBot(account_names=["account"])
stop_flag, pause_flag = bot.get_flags("account")

while not stop_flag.is_set():
    if pause_flag.is_set():
        time.sleep(1)
        continue

    # 執行監控 / 策略程式邏輯
    execute_strategy()
```

---

## 🧱 模組依賴

- `python-telegram-bot==13.15`
- 需設定：
  - `NotifyConfig.TELEGRAM_TOKEN`（Telegram Bot 的 Token）
  - `NotifyConfig.TELEGRAM_CHAT_ID`（允許控制的使用者 chat_id）

## 💬 指令範例與行為說明（文字訊息模式）

⚠️ 僅當 `handle_text()` 被啟用且 `chat_id` 通過白名單驗證時有效。

---

### 1. ✅ 暫停交易

**輸入訊息：**
```
/pause account
```

**機器人回覆：**
```
🛑 [<name>] 已暫停監控
```

---

### 2. ✅ 繼續交易

**輸入訊息：**
```
/resume account
```

**機器人回覆：**
```
✅ [<name>] 已恢復監控
```

---

### 3. ✅ 停止交易

**輸入訊息：**
```
/stop account
```

**機器人回覆：**
```
❌ [<name>] 程式即將停止
```

---

### 4. ✅ 查詢監控狀態

**輸入訊息：**
```
/status account
```

**依據目前旗標狀態，回覆以下三種之一：**
- `[<name>] 目前狀態：✅ 交易中`
- `[<name>] 目前狀態：🛑 暫停交易中`
- `[<name>] 目前狀態：❌ 已關閉`

---

### 5. ✅ 查詢持倉部位

**輸入訊息：**
```
/position account
```

**機器人回覆：**
```python
📦 [<name>] 持倉：{'TXF': 2, 'MTX': 5}
```

---

### ❓ 其他非預期文字

**輸入訊息：**
```
/-----
```

**回覆內容：**
```
❓ 請使用正確指令，如：/pause account
```

