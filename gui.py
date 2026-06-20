import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from trader.config import ACCOUNTS, StrategyList
from trader.utils import runtime
from trader.utils.database import db
from trader.utils.database.tables import SecurityInfo, UserSettings

try:
    from trader.scripts.StrategySet.期貨無腦多 import MaxQuantityList
except Exception:
    MaxQuantityList = None


LOG_DIR = Path("logs")
RUN_SCRIPT = Path("run.py")
STATUS_LABELS = {
    "running": "🟢 交易中",
    "paused": "🟡 暫停中",
    "starting": "🟡 啟動中",
    "stopping": "🟡 停止中",
    "stopped": "🔴 已停止",
    "error": "🔴 異常",
}


st.set_page_config(
    page_title="AutoTradingPlatform",
    layout="wide",
)


def init_state():
    if "processes" not in st.session_state:
        st.session_state.processes = {}


def query_user_settings() -> pd.DataFrame:
    return db.query(UserSettings)


def account_options() -> list[str]:
    accounts = set(ACCOUNTS or [])
    settings = query_user_settings()
    if not settings.empty and "account" in settings:
        accounts.update(settings.account.dropna().astype(str).tolist())
    return sorted(accounts)


def process_for(account: str):
    process = st.session_state.processes.get(account)
    if process and process.poll() is None:
        return process
    if process:
        st.session_state.processes.pop(account, None)
    return None


def is_running(account: str) -> bool:
    return process_for(account) is not None


def parse_time(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def is_status_fresh(status: dict, max_age_seconds: int = 60) -> bool:
    updated_at = parse_time(status.get("updated_at"))
    if not updated_at:
        return False
    return (datetime.now() - updated_at).total_seconds() <= max_age_seconds


def is_account_active(account: str) -> bool:
    status = runtime.read_status(account)
    if status.get("status") in {"running", "paused", "starting", "stopping"}:
        return is_status_fresh(status)
    return is_running(account)


def session_log_path(account: str) -> Path:
    return LOG_DIR / f"{account}.session.log"


def start_trader(account: str):
    if is_account_active(account):
        return False, f"{account} 已經在執行"

    LOG_DIR.mkdir(exist_ok=True)
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "-TASK",
        "auto_trader",
        "-ACCT",
        account,
    ]
    popen_kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }

    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    out_path = session_log_path(account)
    with out_path.open("a", encoding="utf-8", errors="ignore") as out:
        popen_kwargs["stdout"] = out
        process = subprocess.Popen(command, **popen_kwargs)

    st.session_state.processes[account] = process
    runtime.write_status(
        account,
        "starting",
        "GUI 已啟動交易 process",
        pid=process.pid,
    )
    runtime.write_session(
        account,
        pid=process.pid,
        command=" ".join(command),
        cwd=str(Path.cwd()),
        session_log=str(out_path),
        started_at=runtime.now_text(),
        launcher="gui",
    )
    return True, f"{account} 交易 process 已啟動"


def send_command(account: str, command: str, payload: dict | None = None):
    runtime.write_command(account, command, payload)


def status_text(account: str) -> tuple[str, str]:
    status = runtime.read_status(account)
    state = status.get("status")
    if not state:
        state = "running" if is_account_active(account) else "stopped"

    label = STATUS_LABELS.get(state, state)
    updated_at = status.get("updated_at", "")
    message = status.get("message", "")
    details = " | ".join(x for x in [updated_at, message] if x)
    return label, details


def log_path(account: str) -> Path:
    return LOG_DIR / f"{account}.log"


def latest_log(account: str, limit: int = 300) -> list[str]:
    return runtime.tail_lines(log_path(account), limit=limit)


def query_positions(account: str) -> pd.DataFrame:
    try:
        return db.query(SecurityInfo, SecurityInfo.account == account)
    except Exception:
        return pd.DataFrame()


def query_max_qty(account: str) -> pd.DataFrame:
    if MaxQuantityList is None:
        return pd.DataFrame()
    try:
        return db.query(MaxQuantityList, MaxQuantityList.account == account)
    except Exception:
        return pd.DataFrame()


def update_max_qty_table(account: str, strategy: str, market: str, target: str, max_qty: int):
    if MaxQuantityList is None:
        return False

    condition = MaxQuantityList.account == account
    condition &= MaxQuantityList.strategy == strategy
    condition &= MaxQuantityList.target == target
    current = db.query(MaxQuantityList, condition)

    if current.empty:
        db.add_data(
            MaxQuantityList,
            account=account,
            strategy=strategy,
            market=market,
            target=target,
            max_quantity=max_qty,
        )
    else:
        db.update(
            MaxQuantityList,
            {"market": market, "max_quantity": max_qty},
            condition,
        )

    return True


def update_user_settings(account: str, values: dict):
    values = {k: v for k, v in values.items() if v is not None}
    if db.check_exist(UserSettings, account=account):
        db.update(UserSettings, values, UserSettings.account == account)
    else:
        db.add_data(UserSettings, **values)


def delete_user_settings(account: str):
    db.delete(UserSettings, UserSettings.account == account)


def user_settings_form(prefix: str, defaults: dict | None = None):
    defaults = defaults or {}
    col1, col2 = st.columns(2)
    with col1:
        account = st.text_input(
            "帳戶代號",
            value=str(defaults.get("account", "")),
            key=f"{prefix}_account",
        )
        api_key = st.text_input(
            "API 金鑰",
            value=str(defaults.get("api_key", "")),
            type="password",
            key=f"{prefix}_api_key",
        )
        account_id = st.text_input(
            "帳戶 ID",
            value=str(defaults.get("account_id", "")),
            key=f"{prefix}_account_id",
        )
        mode = st.selectbox(
            "交易模式",
            ["Simulation", "All"],
            index=0 if defaults.get("mode", "Simulation") != "All" else 1,
            key=f"{prefix}_mode",
        )
        trading_period = st.selectbox(
            "交易時段",
            ["Day", "Night", "Both"],
            index=["Day", "Night", "Both"].index(
                defaults.get("trading_period", "Day")
                if defaults.get("trading_period", "Day") in ["Day", "Night", "Both"]
                else "Day"
            ),
            key=f"{prefix}_trading_period",
        )

    with col2:
        secret_key = st.text_input(
            "API 密鑰",
            value=str(defaults.get("secret_key", "")),
            type="password",
            key=f"{prefix}_secret_key",
        )
        ca_passwd = st.text_input(
            "交易憑證密碼",
            value=str(defaults.get("ca_passwd", "")),
            type="password",
            key=f"{prefix}_ca_passwd",
        )
        init_balance = st.number_input(
            "帳戶起始資金",
            min_value=0,
            value=int(defaults.get("init_balance", 0) or 0),
            step=10000,
            key=f"{prefix}_init_balance",
        )
        marging_trading_amount = st.number_input(
            "融資額度",
            min_value=0,
            value=int(defaults.get("marging_trading_amount", 0) or 0),
            step=10000,
            key=f"{prefix}_marging_trading_amount",
        )
        short_selling_amount = st.number_input(
            "融券額度",
            min_value=0,
            value=int(defaults.get("short_selling_amount", 0) or 0),
            step=10000,
            key=f"{prefix}_short_selling_amount",
        )
        margin_amount = st.number_input(
            "保證金額度",
            min_value=0,
            value=int(defaults.get("margin_amount", 0) or 0),
            step=10000,
            key=f"{prefix}_margin_amount",
        )

    return {
        "account": account.strip(),
        "api_key": api_key,
        "secret_key": secret_key,
        "account_id": account_id,
        "ca_passwd": ca_passwd,
        "mode": mode,
        "init_balance": int(init_balance),
        "marging_trading_amount": int(marging_trading_amount),
        "short_selling_amount": int(short_selling_amount),
        "trading_period": trading_period,
        "margin_amount": int(margin_amount),
    }


def render_control_panel(selected_account: str):
    label, details = status_text(selected_account)
    active = is_account_active(selected_account)
    process = process_for(selected_account)
    status = runtime.read_status(selected_account)
    session = runtime.read_session(selected_account)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("監控狀態", label)
    col2.metric("Heartbeat", "有效" if active else "無")
    col3.metric("帳號", selected_account)
    col4.metric("Log", log_path(selected_account).name)
    if details:
        st.caption(details)

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("啟動", disabled=active):
        ok, msg = start_trader(selected_account)
        st.success(msg) if ok else st.info(msg)
    if c2.button("暫停"):
        send_command(selected_account, "pause")
        st.info("已送出暫停指令")
    if c3.button("恢復"):
        send_command(selected_account, "resume")
        st.info("已送出恢復指令")
    if c4.button("停止"):
        send_command(selected_account, "stop")
        st.warning("已送出停止指令")

    st.subheader("Session")
    s1, s2, s3 = st.columns(3)
    s1.metric("PID", status.get("pid") or session.get("pid") or "")
    s2.metric(
        "GUI handle",
        "持有" if process and process.poll() is None else "未持有",
    )
    s3.metric("啟動來源", session.get("launcher", "bat/runtime"))
    st.text_input("啟動時間", value=session.get("started_at", ""), disabled=True)
    st.text_input(
        "工作目錄",
        value=session.get("cwd", str(Path.cwd())),
        disabled=True,
    )
    st.text_input("啟動命令", value=session.get("command", ""), disabled=True)
    st.text_input(
        "Session log",
        value=session.get("session_log", str(
            session_log_path(selected_account))),
        disabled=True,
    )

    st.subheader("最近交易 log")
    st.code("\n".join(latest_log(selected_account, limit=80)), language="text")


def render_dashboard(selected_account: str):
    lines = latest_log(selected_account, limit=500)
    indicator = runtime.parse_latest_indicator(lines)

    if indicator:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Open/Close", indicator.get("side", ""))
        col2.metric("當前指數", indicator.get("price", ""))
        col3.metric("ATR", indicator.get("atr", ""))
        col4.metric("STD15", indicator.get("std15", ""))
        col5.metric("DTP", indicator.get("dtp", ""))
        st.caption(indicator.get("raw", ""))
    else:
        st.info("尚未在 log 中找到 [Open] 或 [Close] 指標輸出。")

    positions = query_positions(selected_account)
    st.subheader("目前持倉")
    if positions.empty:
        st.info("目前沒有持倉資料。")
    else:
        st.dataframe(
            positions,
            use_container_width=True,
            hide_index=True,
            key=f"{selected_account}_positions",
        )


def render_max_qty_table(selected_account: str):
    max_qty = query_max_qty(selected_account)
    if max_qty.empty:
        st.info("目前沒有 max_quantity_list 資料，或資料表尚未建立。")
    else:
        st.dataframe(
            max_qty,
            use_container_width=True,
            hide_index=True,
            key=f"{selected_account}_max_qty",
        )


init_state()

st.title("AutoTradingPlatform")

accounts = account_options()
if not accounts:
    st.warning("尚未找到帳號設定，請先在使用者專區新增 UserSettings。")
    selected_account = ""
else:
    selected_account = st.sidebar.selectbox("交易帳號", accounts)

if st.sidebar.button("Refresh"):
    st.rerun()

refresh_mode = st.sidebar.radio(
    "刷新模式",
    ["每隔 t 秒自動刷新", "不自動刷新"],
)
refresh_seconds = st.sidebar.slider(
    "t 秒",
    min_value=2,
    max_value=60,
    value=5,
    disabled=refresh_mode == "不自動刷新",
)
refresh_interval = (
    f"{refresh_seconds}s"
    if refresh_mode == "每隔 t 秒自動刷新"
    else None
)

tab_control, tab_dashboard, tab_strategy, tab_users = st.tabs(
    ["交易控制台", "監控 Dashboard", "策略 max_qty", "使用者專區"]
)

with tab_control:
    if selected_account:
        st.fragment(
            render_control_panel,
            run_every=refresh_interval,
        )(selected_account)

with tab_dashboard:
    if selected_account:
        st.fragment(
            render_dashboard,
            run_every=refresh_interval,
        )(selected_account)

with tab_strategy:
    if selected_account:
        st.subheader("max_qty 查詢與更新")
        st.fragment(
            render_max_qty_table,
            run_every=refresh_interval,
        )(selected_account)

        max_qty = query_max_qty(selected_account)
        strategies = StrategyList.All or (
            sorted(max_qty.strategy.unique().tolist())
            if not max_qty.empty
            else []
        )
        with st.form("max_qty_form"):
            col1, col2, col3, col4 = st.columns(4)
            strategy = col1.selectbox(
                "策略", strategies) if strategies else col1.text_input("策略")
            market = col2.selectbox("市場", ["Futures", "Stocks", "Options"])
            target = col3.text_input("標的")
            value = col4.number_input("max_qty", min_value=0, value=0, step=1)
            submitted = st.form_submit_button("更新 max_qty")

        if submitted:
            if not strategy or not target:
                st.error("請輸入策略與標的。")
            else:
                ok = update_max_qty_table(
                    selected_account,
                    strategy,
                    market,
                    target.strip(),
                    int(value),
                )
                if ok:
                    send_command(
                        selected_account,
                        "update_max_qty",
                        {
                            "strategy": strategy,
                            "target": target.strip(),
                            "max_qty": int(value),
                        },
                    )
                    st.success("已更新 DB，並送出指令；若交易 process 正在跑，會在下一個監控圈同步。")
                else:
                    st.error("max_quantity_list 資料表尚未可用。")

with tab_users:
    st.subheader("UserSettings")
    settings = query_user_settings()
    if settings.empty:
        st.info("目前沒有 UserSettings 資料。")
    else:
        display = settings.copy()
        for col in ["api_key", "secret_key", "ca_passwd"]:
            if col in display:
                display[col] = display[col].apply(
                    lambda x: "" if not x else "********")
        st.dataframe(display, use_container_width=True, hide_index=True)

    user_accounts = (
        settings.account.dropna().astype(str).tolist()
        if not settings.empty and "account" in settings
        else []
    )
    selected_user = st.selectbox(
        "選擇要修改或刪除的帳號",
        ["新增"] + user_accounts,
    )

    defaults = {}
    if selected_user != "新增" and not settings.empty:
        row = settings[settings.account == selected_user]
        if not row.empty:
            defaults = row.iloc[0].to_dict()

    with st.form("user_settings_form"):
        values = user_settings_form("user_settings", defaults)
        col_save, col_delete = st.columns(2)
        save_user = col_save.form_submit_button("儲存")
        delete_user = col_delete.form_submit_button(
            "刪除",
            disabled=selected_user == "新增",
        )

    if save_user:
        if not values["account"]:
            st.error("帳戶代號不可為空。")
        else:
            update_user_settings(values["account"], values)
            st.success(f"{values['account']} 已儲存")
            st.rerun()

    if delete_user:
        delete_user_settings(selected_user)
        st.warning(f"{selected_user} 已刪除")
        st.rerun()
