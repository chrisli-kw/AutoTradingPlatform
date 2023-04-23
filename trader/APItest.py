import time
import shioaji as sj
from dotenv import dotenv_values

from . import TODAY
from .utils.time import TimeTool


class APITester(TimeTool):
    '''API串接測試，正式下單前，必須先經過測試，才可開通下單功能'''

    def simulation_test(self, API_KEY, SECRET_KEY, acct):
        api = sj.Shioaji(simulation=True)
        api.login(API_KEY, SECRET_KEY)

        is_simulate = api.simulation
        if is_simulate:
            print(f"Log in to {acct} with simulation mode: {is_simulate}")
            time.sleep(30)
        else:
            print(f"Log in to {acct} with real mode, log out")
            api.logout()
            return

        # 股票下單測試
        stockid = '2603'
        contract = api.Contracts.Stocks[stockid]
        order = api.Order(
            action=sj.constant.Action.Buy,
            price=18,
            quantity=1,
            price_type=sj.constant.StockPriceType.LMT,
            order_type=sj.constant.OrderType.ROD,
            order_lot=sj.constant.StockOrderLot.Common,

            account=api.stock_account
        )
        print(f'Stock order content:\n{order}')

        print('Place stock order')
        trade = api.place_order(contract, order, timeout=0)
        print(f'Done:\n{trade}')
        time.sleep(2)

        # 期貨下單測試
        futuresid = f'MXF{self.GetDueMonth(TODAY)}'
        contract = api.Contracts.Futures.MXF[futuresid]
        order = api.Order(
            action=sj.constant.Action.Buy,
            price=15000,
            quantity=1,
            price_type=sj.constant.FuturesPriceType.LMT,
            order_type=sj.constant.OrderType.ROD,
            octype=sj.constant.FuturesOCType.Auto,
            account=api.futopt_account,
        )
        print(f'Futures order content:\n{order}')
        print('Place futures order')
        trade = api.place_order(contract, order, timeout=0)
        print(f'Done:\n{trade}')

        print(f'Log out {acct}: {api.logout()}\n')

    def verify_test(self, API_KEY, SECRET_KEY, acct):
        api = sj.Shioaji(simulation=False)
        api.login(API_KEY, SECRET_KEY)
        print(f"Log in to {acct} with real mode")
        time.sleep(10)

        accounts = api.list_accounts()
        for acct in accounts:
            print(f'Account {acct.account_id}: {acct.signed}')

        print(f'Log out {acct}: {api.logout()}\n')

    def run(self, account):
        '''Shioaji 帳號測試'''

        for i in range(2):
            config = dotenv_values(f'./lib/envs/{account}.env')
            API_KEY = config['API_KEY']
            SECRET_KEY = config['SECRET_KEY']

            if i == 0:
                self.simulation_test(API_KEY, SECRET_KEY, account)
                self.CountDown(720)
            else:
                self.verify_test(API_KEY, SECRET_KEY, account)


if __name__ == "__main__":
    tester = APITester()

    account = 'test_account_1'
    tester.run(account)
