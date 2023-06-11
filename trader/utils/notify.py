import logging
import requests
import pandas as pd
from .. import API, PATH
from ..config import TOKEN_INFO, TOKEN_MONITOR


class Notification:
    NotifyURL = "https://notify-api.line.me/api/notify"
    order_cond = {
        'Cash': '現股',
        'MarginTrading': '融資',
        'ShortSelling': '融券'
    }
    order_lot = {
        'Common': '張',
        'IntradayOdd': '股'
    }
    oc_type = {
        'New': '新倉',
        'Cover': '平倉'
    }

    def headers(self, msgType: str):
        '''LINE notify推播header設定'''
        if msgType in ['Order', 'Deal', 'AccountInfo', 'Monitor', 'Tasker', 'Crawler']:
            return {"Authorization": f"Bearer {TOKEN_MONITOR}"} if TOKEN_MONITOR else {}
        return {"Authorization": f"Bearer {TOKEN_INFO}"} if TOKEN_INFO else {}

    def post(self, message: str, image_name=None, msgType: str = 'price'):
        '''Line Notify 推播，可傳送文字或圖片訊息'''

        data = {'message': message}
        headers = self.headers(msgType)

        if not headers:
            return

        if image_name:
            image = open(f'{PATH}/{image_name}.jpg', 'rb')
            files = {'imageFile': image}
            requests.post(
                self.NotifyURL,
                headers=headers,
                data=data,
                files=files
            )
        else:
            requests.post(self.NotifyURL, headers=headers, data=data)

    def post_tftOrder(self, stat, msg: dict):
        '''發送推播-股票委託'''
        logging.debug(f'[{stat}][{msg}]')

        stock = msg['contract']['code']
        name = API.Contracts.Stocks[stock].name
        order = msg['order']
        account = order['account']['account_id']
        cond = self.order_cond[order['order_cond']]
        lot = self.order_lot[order['order_lot']]

        operation = msg['operation']
        if operation['op_code'] == '00' or operation['op_msg'] == '':
            text = f"\n【委託成功】{name}-{stock}\n【帳號】{account}\n【{cond}】{order['action']} {order['quantity']}{lot}"
            self.post(text, msgType='Order')

        if operation['op_code'] == '88':
            text = f"\n【委託失敗】{name}-{stock}\n【帳號】{account}\n【{operation['op_msg']}】"
            self.post(text, msgType='Order')

        if operation['op_type'] == 'Cancel':
            text = f"\n【刪單成功】{name}-{stock}\n【帳號】{account}\n【{cond}】{order['action']} {order['quantity']}{lot}"
            self.post(text, msgType='Order')

    def post_tftDeal(self, stat, msg: dict):
        '''發送推播-股票成交'''
        logging.debug(f'[{stat}][{msg}]')

        stock = msg['code']
        name = API.Contracts.Stocks[stock].name
        account = msg['account_id']
        cond = self.order_cond[msg['order_cond']]
        lot = self.order_lot[msg['order_lot']]
        price = msg['price']
        text = f"\n【成交】{name}-{stock}\n【帳號】{account}\n【{cond}】{msg['action']} {msg['quantity']}{lot} {price}元"
        self.post(text, msgType='Deal')

    def post_fOrder(self, stat, msg: dict):
        '''發送推播-期貨委託'''
        logging.debug(f'[{stat}][{msg}]')

        code = msg['contract']['code']
        delivery_month = msg['contract']['delivery_month']
        name = API.Contracts['Futures'][code][code+delivery_month].name
        order = msg['order']
        account = order['account']['account_id']
        oc_type = self.oc_type[order['oc_type']]
        quantity = order['quantity']

        operation = msg['operation']
        if operation['op_code'] == '00' or operation['op_msg'] == '':
            text = f"\n【委託成功】{name}({code+delivery_month})\n【帳號】{account}\n【{oc_type}】{order['action']} {quantity}口"
            self.post(text, msgType='Order')

        if operation['op_code'] != '00':
            text = f"\n【委託失敗】{name}({code+delivery_month})\n【帳號】{account}\n【{operation['op_msg']}】"
            self.post(text, msgType='Order')

    def post_fDeal(self, stat, msg: dict):
        '''發送推播-期貨成交'''
        logging.debug(f'[{stat}][{msg}]')

        code = msg['code']
        delivery_month = msg['delivery_month']
        name = API.Contracts['Futures'][code][code+delivery_month].name
        account = msg['account_id']
        price = msg['price']
        quantity = msg['quantity']
        text = f"\n【成交】{name}({code+delivery_month})\n【帳號】{account}\n【{msg['action']}】{quantity}口 {price}元"
        self.post(text, msgType='Deal')

    def post_put_call_ratio(self, df_pcr: pd.DataFrame):
        '''發送推播-Put/Call Ratio'''

        if df_pcr.shape[0]:
            put_call_ratio = df_pcr.PutCallRatio.values[0]
        else:
            put_call_ratio = '查無資料'

        text = f"\n【本日Put/Call Ratio】 {put_call_ratio}"
        self.post(text, msgType='Msg')

    def post_account_info(self, account_id: str, info: dict):
        '''發送推播-每日帳務'''

        if info:
            text = f'\n帳號: {account_id}'
            for k, i in info.items():
                text += f'\n{k}: {i}'
        else:
            text = '查無資訊'

        text = f"\n【盤後帳務資訊】{text}"
        self.post(text, msgType='AccountInfo')

    def post_stock_selection(self, df: pd.DataFrame):
        '''發送推播-每日選股清單'''

        if df.shape[0]:
            strategies = df.Strategy.unique()

            text = ''
            for s in strategies:
                temp = df[df.Strategy == s]
                temp = temp.set_index('company_name').name.to_dict()

                _text = ''
                for k, v in temp.items():
                    _text += f'\n{k}({v}), '

                text += f"\n----------{s}----------{_text.rstrip(', ')}\n"
            text = text.rstrip('\n')

        else:
            text = '無'

        text = f"\n【本日選股清單】{text}"
        self.post(text, msgType='StockSelect')
