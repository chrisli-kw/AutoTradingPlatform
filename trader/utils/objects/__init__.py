from collections import namedtuple


class Margin:
    available_margin = 0
    equity = 0


Action = namedtuple(
    typename="Action",
    field_names=['position', 'reason', 'msg', 'price', 'action'],
    defaults=[0, '', '', 0, '']
)
