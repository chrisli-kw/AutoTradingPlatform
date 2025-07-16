from flask import Flask, request, render_template

from trader.utils.objects.env import UserEnv
from trader.utils.database import db
from trader.utils.database.tables import UserSettings


def _create_env(inputs):
    modes = {
        '作多且做空, 且可買進也可賣出': 'All',
        '模擬': 'Simulation'
    }

    account_name = inputs['ACCOUNT']

    users = db.query(UserSettings, UserSettings.account == account_name)
    if users.empty:
        nth_account = 1
    else:
        nth_account = users.shape[0] + 1

    account_name = f'{account_name}_{nth_account}'
    env = {
        'account': account_name,
        'api_key': inputs['API_KEY'],
        'secret_key': inputs['SECRET_KEY'],
        'account_id': inputs['ACCOUNT_ID'],
        'ca_passwd': inputs['CA_PASSWD'],
        'mode': modes[inputs['MODE']],
        'init_balance': 100000,
        'marging_trading_amount': inputs['MARGING_TRADING_AMOUNT'],
        'short_selling_amount': inputs['SHORT_SELLING_AMOUNT'],
        'trading_period': inputs['TRADING_PERIOD'],
        'margin_amount': inputs['MARGIN_AMOUNT']
    }
    UserEnv(account_name).env_to_db(**env)

    return {account_name: env}


def _create_bat(account_name):
    myBat = open(f'./lib/schedules/auto_trader_{account_name}.bat', 'w+')
    myBat.write(
        f"""
        cd /d %~dp0/../../..
        set SJ_LOG_PATH=%~dp0/../../../logs/shioaji.log
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
