from flask import Flask, request, render_template

from trader import file_handler


def _create_env(inputs):
    modes = {
        '作多只賣出': 'LongSell',
        '作多只買進': 'LongBuy',
        '作多買進&賣出': 'LongBoth',
        '作空只賣出': 'ShortSell',
        '作空只買進': 'ShortBuy',
        '作空買進&賣出': 'ShortBoth',
        '作多且做空, 且可買進也可賣出': 'All',
        '模擬': 'Simulation'
    }
    markets = {
        '只有股票': 'stock',
        '只有期貨': 'futures',
        '股票 & 期貨': 'stock and futures',
    }

    account_name = inputs['ACCOUNT_NAME']

    users = file_handler.listdir('./lib/envs', account_name)
    if users:
        nth_account = max(users) + 1
    else:
        nth_account = 1

    account_name = f'{account_name}_{nth_account}'
    myBat = open(f"./lib/envs/{account_name}.env", 'w+', encoding='utf-8')

    content = f"""# 使用者資訊
        ACCOUNT_NAME={account_name}
        API_KEY = {inputs['API_KEY']}
        SECRET_KEY = {inputs['SECRET_KEY']}
        ACCOUNT_ID = {inputs['ACCOUNT_ID']}
        CA_PASSWD = {inputs['CA_PASSWD']}

        # 崩盤日
        KBAR_START_DAYay=

        # 多空模式 (英文)
        # 作多: LongSell(只賣出), LongBuy(只買進), LongBoth(買進&賣出)
        # 作空: ShortSell(只賣出), ShortBuy(只買進), ShortBoth(買進&賣出)
        # All: 作多且做空, 且可買進也可賣出
        # Simulation: 可做策略監控(多 or 空 or 多策略), 但不下單
        MODE={modes[inputs['MODE']]}

        # 市場類型:stock/futures/stock and futures
        MARKET={markets[inputs['MARKET']]}

        # 加入要監控的清單，沒有等號後面空白
        FILTER_IN={inputs['FILTER_IN']}

        # 過濾掉不監控的清單，沒有等號後面空白
        FILTER_OUT={inputs['FILTER_OUT']}

        # 股票要執行的策略
        STRATEGY_STOCK={inputs['STRATEGY_STOCK']}

        # 高價股門檻
        PRICE_THRESHOLD=99999

        # 起始部位
        INIT_POSITION=100000

        # 部位上限/可委託金額上限
        POSITION_LIMIT_LONG={inputs['POSITION_LIMIT_LONG']}
        POSITION_LIMIT_SHORT={inputs['POSITION_LIMIT_SHORT']}

        # 投資組合股票數上限
        N_LIMIT_LS={inputs['N_LIMIT_LS']}
        N_LIMIT_SS={inputs['N_LIMIT_SS']}

        # 投資組合股票數上限類型: 固定(constant)/隨大盤環境波動(float)
        N_STOCK_LIMIT_TYPE={inputs['N_STOCK_LIMIT_TYPE']}

        # 每次買進的單位(張)
        BUY_UNIT={inputs['BUY_UNIT']}

        # 買進的單位數類型: 固定(constant)/隨淨值等比例增加(float)
        BUY_UNIT_TYPE={inputs['BUY_UNIT_TYPE']}

        # 現股/融資 (Cash/MarginTrading/ShortSelling)
        ORDER_COND1={inputs['ORDER_COND1']}
        ORDER_COND2={inputs['ORDER_COND2']}

        # 最長持有天數(含六日)
        HOLD_DAY={inputs['HOLD_DAY']}

        # ----------------------------------- 期貨設定 ----------------------------------- #
        # 交易時段(Day/Night/Both)
        TRADING_PERIOD={inputs['TRADING_PERIOD']}

        # 期貨要執行的策略
        STRATEGY_FUTURES={inputs['STRATEGY_FUTURES']}

        # 可下單的保證金額上限
        MARGIN_LIMIT = {inputs['MARGIN_LIMIT']}

        # 投資組合期權數上限
        N_FUTURES_LIMIT={inputs['N_FUTURES_LIMIT']}

        # 期權投資組合期權數上限類型: 固定(constant)/隨大盤環境波動(float)
        N_FUTURES_LIMIT_TYPE={inputs['N_FUTURES_LIMIT_TYPE']}

        # 每次期權建倉的單位(口)
        N_SLOT={inputs['N_SLOT']}

        # 建倉的口數類型: 固定(constant)/隨淨值等比例增加(float)
        N_SLOT_TYPE={inputs['N_SLOT_TYPE']}
        """.replace('        ', '')

    myBat.write(content)
    myBat.close()
    return {account_name: content}


def _create_bat(account_name):
    myBat = open(f'./lib/schedules/auto_trader_{account_name}.bat', 'w+')
    myBat.write(
        f"""call C:/Users/%username%/anaconda3/Scripts/activate.bat

        cd /d %~dp0/../..

        set SJ_LOG_PATH=%~dp0/../../logs/shioaji.log

        python run.py -TASK auto_trader -ACCT {account_name}
        """.replace('        ', ''))

    myBat.close()


def bat_add_script(name):
    filename = './lib/schedules/auto_trader_all.bat'
    myBat = open(filename, 'r')
    contents = myBat.readlines()
    command_line = f'start /min cmd /c %~dp0/auto_trader_{name}.bat'
    if command_line not in contents:
        contents.append(f'''\n\n{command_line}''')

    myBat = open(filename, 'w')
    myBat.write("".join(contents))
    myBat.close()


app = Flask(__name__)


# Homepage
@ app.route("/", methods=['GET'])
def account():
    return render_template('index.html')


@ app.route("/account-result", methods=['POST'])
def account_result():

    data = request.form
    result = _create_env(data)

    # # 新增個人bat檔
    account_name = list(result)[0]
    _create_bat(account_name)

    # # 加入個人bat檔執行指令
    # bat_add_script(name)

    result = {'msg': '新增成功'}
    return render_template('account_result.html', result=result)
