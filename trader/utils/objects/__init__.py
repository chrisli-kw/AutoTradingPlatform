from collections import namedtuple


class Margin:
    available_margin = 0
    equity = 0


Action = namedtuple(
    typename="Action",
    field_names=['action', 'reason', 'quantity', 'isRaiseQty'],
    defaults=['', 0, '', False]
)
