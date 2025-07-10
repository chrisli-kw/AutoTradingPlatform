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

    account_name = inputs['ACCOUNT_NAME']

    users = file_handler.Operate.listdir('./lib/envs', account_name)
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

        # 多空模式 (英文)
        # 作多: LongSell(只賣出), LongBuy(只買進), LongBoth(買進&賣出)
        # 作空: ShortSell(只賣出), ShortBuy(只買進), ShortBoth(買進&賣出)
        # All: 作多且做空, 且可買進也可賣出
        # Simulation: 可做策略監控(多 or 空 or 多策略), 但不下單
        MODE={modes[inputs['MODE']]}

        # 起始部位
        INIT_POSITION=100000

        # 部位上限/可委託金額上限
        POSITION_LIMIT_LONG={inputs['POSITION_LIMIT_LONG']}
        POSITION_LIMIT_SHORT={inputs['POSITION_LIMIT_SHORT']}

        # 現股/融資 (Cash/MarginTrading/ShortSelling)
        ORDER_COND1={inputs['ORDER_COND1']}
        ORDER_COND2={inputs['ORDER_COND2']}

        # ----------------------------------- 期貨設定 ----------------------------------- #
        # 交易時段(Day/Night/Both)
        TRADING_PERIOD={inputs['TRADING_PERIOD']}

        # 可下單的保證金額上限
        MARGIN_LIMIT = {inputs['MARGIN_LIMIT']}

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
@app.route("/", methods=['GET'])
def account():
    return render_template('index.html')


@app.route("/account-result", methods=['POST'])
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
