import time
from shioaji import constant

from .config import TODAY, create_api
from .utils.time import time_tool
from .utils.objects.env import UserEnv


class APITester:
    '''API串接測試，正式下單前，必須先經過測試，才可開通下單功能'''

    def __init__(self):
        self.api = create_api(simulation=True)

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
            action=constant.Action.Buy,
            price=contract.limit_up,
            quantity=1,
            price_type=constant.StockPriceType.LMT,
            order_type=constant.OrderType.ROD,
            order_lot=constant.StockOrderLot.Common,

            account=self.api.stock_account
        )
        print(f'\n[stock order] content: {order}')

        print('[stock order] Place order')
        trade = self.api.place_order(contract, order, timeout=0)
        print(f'[stock order] Done: {trade}')
        time.sleep(2)

        # 期貨下單測試
        futuresid = f'MXF{time_tool.GetDueMonth(TODAY)}'
        contract = self.api.Contracts.Futures.MXF[futuresid]
        order = self.api.Order(
            action=constant.Action.Buy,
            price=contract.limit_up,
            quantity=1,
            price_type=constant.FuturesPriceType.LMT,
            order_type=constant.OrderType.ROD,
            octype=constant.FuturesOCType.Auto,
            account=self.api.futopt_account,
        )
        print(f'\n[futures order] content: {order}')
        print('[futures order] Place order')
        trade = self.api.place_order(contract, order, timeout=0)
        print(f'[futures order] Done: {trade}')

        print(f'\nLog out {acct}: {self.api.logout()}\n')

    def verify_test(self, env):
        self.api = create_api(simulation=False)
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
                time_tool.CountDown(720)
            else:
                self.verify_test(config)


if __name__ == "__main__":
    tester = APITester()

    account = 'test_account_1'
    tester.run(account)
