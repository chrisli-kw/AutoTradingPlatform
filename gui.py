
import streamlit as st

from run import logging
from trader.tasker import get_tasks
from trader.config import API


# Session states
if 'api' not in st.session_state:
    st.session_state.api = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'log_output' not in st.session_state:
    st.session_state.log_output = ""

st.title("💹 AutoTradingPlatform 自動交易平台")

Tasks = get_tasks()
strategy_option = st.selectbox(
    "選擇要執行的動作",
    options=list(Tasks.keys()),
    key="task"
)
account = st.text_input("主交易帳號:", type="default")

col1, col2, col3 = st.columns(3)
with col1:
    btn_start = st.button("開始", key="start_button")
with col2:
    btn_stop = st.button("停止", key="stop_button")
with col3:
    btn_logout = st.button("登出", key="logout_button")


# 動態 log 顯示區
log_area = st.empty()

if account:
    logging.info(f'The trading account is {account}')

    task = st.session_state['task']
    if btn_start and not st.session_state.logged_in:
        functions = Tasks[task]

        for func in functions:
            st.info(f'[{task}] 執行 {func.__name__}')

            if task == 'create_env':
                st.info(f'打開 http://localhost:8090 已設定交易參數')

            st.session_state.logged_in = True
            if task == 'resistence':
                func(stream_output=log_area)
            elif func.__name__ in ['runAutoTrader', 'runCrawlStockData']:
                func(account=account)
            else:
                func()

            # 更新 log_area（每步驟之後都更新）
            log_area.text_area(
                "📋 執行紀錄", st.session_state.log_output, height=400)
        st.info(f'{task} 執行完畢')

    if btn_stop:
        logging.info(f'Stop the task {task},  log out: {API.logout()}')
        st.session_state.logged_in = False

    if btn_logout and st.session_state.logged_in:
        logging.info(f'API log out: {API.logout()}')
        API.logout()
        st.session_state.logged_in = False
else:
    st.info("請先輸入帳號名稱，以執行自動化操作")
