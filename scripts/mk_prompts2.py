import sys
out = sys.argv[1]
sysmsg = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
prose = """The river wound through the valley like a silver thread, carrying with it the scent of pine and wet stone. In the early morning the mist lay thick over the water, and the only sound was the slow turning of the old mill wheel at the edge of the village. Marta walked down to the river every day before sunrise, a habit she had inherited from her grandmother, who had believed that the river listened to anyone patient enough to sit beside it. She carried a small basket of bread and cheese, and a notebook in which she wrote down whatever the river seemed to say that day.
Some mornings the river said nothing at all, and she wrote only the date and the temperature of the water. Other mornings it spoke in long sentences of ripples and eddies, and her notebook filled with careful drawings of currents and the shapes of floating leaves. The villagers thought her strange, but they also brought her their questions, because they had noticed that the days Marta predicted floods were the days the river rose, and the days she predicted calm were the days the fishermen came home with full nets.
One autumn a stranger arrived, a surveyor with a leather case of instruments and a letter from the regional council. He explained that the council planned to build a dam upstream, and that the river would be diverted into a concrete channel. Marta listened without interrupting, then asked him to sit beside her at the water's edge and say nothing for an hour. The surveyor, to his own surprise, agreed. When the hour had passed he stood up slowly, folded his letter, and said that he would need to write a different report."""
code = '''import json
from dataclasses import dataclass, field


@dataclass
class LineItem:
    sku: str
    quantity: int
    unit_price: float

    def line_total(self) -> float:
        return round(self.quantity * self.unit_price, 2)


@dataclass
class Order:
    order_id: int
    customer: str
    items: list = field(default_factory=list)
    discount: float = 0.0

    def add_item(self, item: LineItem) -> None:
        self.items.append(item)

    def calc_total(self) -> float:
        subtotal = sum(item.line_total() for item in self.items)
        return round(subtotal * (1.0 - self.discount), 2)

    def to_json(self) -> str:
        return json.dumps(
            {
                "order_id": self.order_id,
                "customer": self.customer,
                "items": [vars(i) for i in self.items],
                "total": self.calc_total(),
            },
            indent=2,
        )


def summarize(orders: list) -> dict:
    totals = {}
    for order in orders:
        totals[order.customer] = totals.get(order.customer, 0.0) + order.calc_total()
    return {name: round(value, 2) for name, value in totals.items()}


if __name__ == "__main__":
    o = Order(1, "ayse", discount=0.1)
    o.add_item(LineItem("A-100", 2, 19.99))
    o.add_item(LineItem("B-200", 1, 5.5))
    print(o.to_json())
    print(summarize([o]))
'''
p1 = sysmsg + "<|im_start|>user\nRepeat the following text exactly, word for word, but replace every occurrence of the word \"river\" with \"canal\" (and \"River\" with \"Canal\").\n\n" + prose + "<|im_end|>\n<|im_start|>assistant\n"
p2 = sysmsg + "<|im_start|>user\nHere is a Python file:\n```python\n" + code + "```\nOutput the complete file again, but rename the method calc_total to compute_total everywhere. Output only the code.<|im_end|>\n<|im_start|>assistant\n"
open(out + "/p_prose.txt", "w").write(p1)
open(out + "/p_code.txt", "w").write(p2)
