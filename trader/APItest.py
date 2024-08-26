import time
import shioaji as sj

from .config import TODAY
from .utils.time import TimeTool
from .utils.objects.env import UserEnv


class APITester(TimeTool):
    '''API串接測試，正式下單前，必須先經過測試，才可開通下單功能'''

    def __init__(self):
        self.api = sj.Shioaji(simulation=True)

    def simulation_test(self, env):
        acct = env.ACCOUNT_NAME
        self.api.login(env.api_key(), env.secret_key())

        is_simulate = self.api.simulation
        if is_simulate:
            print(f"Log in to {acct} with simulation mode: {is_simulate}")
            time.sleep(30)
        else:
            print(f"Log in to {acct} with real mode, log out")
            self.api.logout()
            return

        # 股票下單測試
        stockid = '2603'
        contract = self.api.Contracts.Stocks[stockid]
        order = self.api.Order(
            action=sj.constant.Action.Buy,
            price=contract.limit_up,
            quantity=1,
            price_type=sj.constant.StockPriceType.LMT,
            order_type=sj.constant.OrderType.ROD,
            order_lot=sj.constant.StockOrderLot.Common,

            account=self.api.stock_account
        )
        print(f'\n[stock order] content: {order}')

        print('[stock order] Place order')
        trade = self.api.place_order(contract, order, timeout=0)
        print(f'[stock order] Done: {trade}')
        time.sleep(2)

        # 期貨下單測試
        futuresid = f'MXF{self.GetDueMonth(TODAY)}'
        contract = self.api.Contracts.Futures.MXF[futuresid]
        order = self.api.Order(
            action=sj.constant.Action.Buy,
            price=contract.limit_up,
            quantity=1,
            price_type=sj.constant.FuturesPriceType.LMT,
            order_type=sj.constant.OrderType.ROD,
            octype=sj.constant.FuturesOCType.Auto,
            account=self.api.futopt_account,
        )
        print(f'\n[futures order] content: {order}')
        print('[futures order] Place order')
        trade = self.api.place_order(contract, order, timeout=0)
        print(f'[futures order] Done: {trade}')

        print(f'\nLog out {acct}: {self.api.logout()}\n')

    def verify_test(self, env):
        self.api = sj.Shioaji(simulation=False)
        self.api.login(env.api_key(), env.secret_key())
        print(f"Log in to {env.ACCOUNT_NAME} with real mode")
        time.sleep(10)

        accounts = self.api.list_accounts()
        for acct in accounts:
            print(f'Account {acct.account_id} signed: {acct.signed}')

        try:
            self.api.list_positions(self.api.stock_account)
        except:
            print('|--------------------------------------|')
            print('|               warning                |')
            print('|  Failed to list stocks positions     |')
            print('|--------------------------------------|')

        try:
            self.api.list_positions(self.api.futopt_account)
        except:
            print('|--------------------------------------|')
            print('|               warning                |')
            print('|  Failed to list futures positions    |')
            print('|--------------------------------------|')

        print(f'Log out {acct}: {self.api.logout()}\n')

    def run(self, account):
        '''Shioaji 帳號測試'''

        for i in range(2):
            config = UserEnv(account)
            if i == 0:
                self.simulation_test(config)
                self.CountDown(720)
            else:
                self.verify_test(config)


if __name__ == "__main__":
    tester = APITester()

    account = 'test_account_1'
    tester.run(account)
