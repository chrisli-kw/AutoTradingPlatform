import time
import shioaji as sj

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
        order = sj.StockOrder(
            action=sj.Action.Buy,
            price=contract.limit_up,
            quantity=1,
            price_type=sj.StockPriceType.LMT,
            order_type=sj.OrderType.ROD,
            order_lot=sj.StockOrderLot.Common,

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
        order = sj.FuturesOrder(
            action=sj.Action.Buy,
            price=contract.limit_up,
            quantity=1,
            price_type=sj.FuturesPriceType.LMT,
            order_type=sj.OrderType.ROD,
            octype=sj.FuturesOCType.Auto,
            account=self.api.futopt_account,
        )
        print(f'\n[futures order] content: {order}')
        print('[futures order] Place order')
        trade = self.api.place_order(contract, order, timeout=0)
        print(f'[futures order] Done: {trade}')

        print(f'\nLog out {acct}: {self.api.logout()}\n')

    def combo_order_test(self, env, buy_code: str, sell_code: str, price: float):
        self.api.login(env.api_key(), env.secret_key())

        leg_buy = self.api.Contracts.Options.TXO[buy_code]
        leg_sell = self.api.Contracts.Options.TXO[sell_code]
        combo = sj.ComboContract(legs=[
            sj.ComboBase.from_contract(leg_buy, action=sj.Action.Buy),
            sj.ComboBase.from_contract(leg_sell, action=sj.Action.Sell),
        ])
        order = sj.ComboOrder(
            price=price,
            quantity=1,
            price_type=sj.FuturesPriceType.LMT,
            order_type=sj.OrderType.IOC,
            octype=sj.FuturesOCType.Auto,
            account=self.api.futopt_account,
        )
        trade = self.api.place_comboorder(combo, order)
        print(f'[combo order] Done: {trade}')

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
