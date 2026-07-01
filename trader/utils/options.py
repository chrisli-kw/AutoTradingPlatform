from dataclasses import dataclass
from typing import Optional

from ..config import API


def normalize_option_type(option_type: str):
    option_type = str(option_type).strip().lower()
    if option_type in ['c', 'call', 'optioncall']:
        return 'C'
    if option_type in ['p', 'put', 'optionput']:
        return 'P'
    raise ValueError(f'Unsupported option_type: {option_type}')


def parse_expiration(expiration: str):
    expiration = str(expiration).strip().upper()
    if 'W' not in expiration:
        return expiration, None

    delivery_month, week = expiration.split('W', 1)
    return delivery_month, week


def iter_option_contracts(api=None):
    if api is None:
        api = API

    for group in api.Contracts.Options:
        for contract in group:
            yield contract


def _contract_option_type(contract):
    option_right = getattr(contract, 'option_right', '')
    value = getattr(option_right, 'value', option_right)
    name = getattr(option_right, 'name', '')

    text = f'{value} {name}'.lower()
    if value == 'C' or 'call' in text:
        return 'C'
    if value == 'P' or 'put' in text:
        return 'P'
    return ''


def _matches_underlying(contract, underlying: str):
    underlying = str(underlying).upper()
    values = [
        getattr(contract, 'category', ''),
        getattr(contract, 'symbol', ''),
        getattr(contract, 'code', ''),
    ]
    return any(str(value).upper().startswith(underlying) for value in values)


def _week_score(contract, week: Optional[str]):
    if week is None:
        return 0

    category = str(getattr(contract, 'category', '')).upper()
    return 0 if category.endswith(str(week)) else 100


def get_option_contract(
        expiration: str,
        strike: float,
        option_type: str,
        underlying: str = 'TX',
        api=None
):
    delivery_month, week = parse_expiration(expiration)
    option_type = normalize_option_type(option_type)
    strike = float(strike)

    matches = []
    for contract in iter_option_contracts(api):
        if not _matches_underlying(contract, underlying):
            continue

        if str(getattr(contract, 'delivery_month', '')) != delivery_month:
            continue

        if float(getattr(contract, 'strike_price', 0)) != strike:
            continue

        if _contract_option_type(contract) != option_type:
            continue

        if week is not None:
            category = str(getattr(contract, 'category', '')).upper()
            if not category.endswith(str(week)):
                continue

        matches.append(contract)

    if not matches:
        raise KeyError(
            f'Option contract not found: {underlying} {expiration} '
            f'{int(strike) if strike.is_integer() else strike} {option_type}'
        )

    matches.sort(key=lambda contract: _week_score(contract, week))
    return matches[0]


def find_option_contract(target: str, api=None):
    target = str(target)
    for contract in iter_option_contracts(api):
        if target in [
            getattr(contract, 'symbol', None),
            getattr(contract, 'code', None),
        ]:
            return contract
    raise KeyError(target)


@dataclass
class OptionLeg:
    action: str
    expiration: str
    strike: float
    option_type: str
    underlying: str = 'TX'


class OptionOrderFactory:
    def __init__(self, order_info_factory, api=None):
        self.OrderInfo = getattr(
            order_info_factory, 'OrderInfo', order_info_factory)
        self.api = api or API

    def contract(
            self,
            expiration: str,
            strike: float,
            option_type: str,
            underlying: str = 'TX'
    ):
        return get_option_contract(
            expiration=expiration,
            strike=strike,
            option_type=option_type,
            underlying=underlying,
            api=self.api
        )

    def order_info(
            self,
            action: str,
            expiration: str,
            strike: float,
            option_type: str,
            quantity: int = 1,
            price: float = 0,
            price_type: str = 'MKT',
            order_type: str = 'IOC',
            octype: str = 'Auto',
            action_type: str = 'Open',
            underlying: str = 'TX',
            order_label: str = '',
            combo_tag: str = '',
            reason: str = ''
    ):
        contract = self.contract(
            expiration=expiration,
            strike=strike,
            option_type=option_type,
            underlying=underlying
        )
        return self.OrderInfo(
            action_type=action_type,
            action=action,
            target=contract.symbol,
            quantity=quantity,
            octype=octype,
            price=price,
            price_type=price_type,
            order_type=order_type,
            order_label=order_label,
            combo_tag=combo_tag,
            reason=reason,
        )

    def combo_order_info(
            self,
            legs: list,
            price: float,
            quantity: int = 1,
            price_type: str = 'LMT',
            order_type: str = 'IOC',
            octype: str = 'Auto',
            action_type: str = 'Open',
            order_label: str = '',
            combo_tag: str = '',
            reason: str = ''
    ):
        combo_legs = []
        symbols = []

        for leg in legs:
            if isinstance(leg, OptionLeg):
                leg = leg.__dict__

            contract = self.contract(
                expiration=leg['expiration'],
                strike=leg['strike'],
                option_type=leg['option_type'],
                underlying=leg.get('underlying', 'TX')
            )
            symbols.append(contract.symbol)
            combo_legs.append({
                'contract': contract,
                'target': contract.symbol,
                'action': leg['action'],
            })

        target = '+'.join(symbols)
        return self.OrderInfo(
            action_type=action_type,
            action=combo_legs[0]['action'],
            target=target,
            quantity=quantity,
            octype=octype,
            price=price,
            price_type=price_type,
            order_type=order_type,
            combo_legs=combo_legs,
            order_label=order_label,
            combo_tag=combo_tag,
            reason=reason,
        )
