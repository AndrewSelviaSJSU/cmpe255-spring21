import matplotlib.pyplot as plt
import pandas as pd


class Solution:
    def __init__(self) -> None:
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep="\t")
    
    def top_x(self, count) -> None:
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        return self.chipo.order_id.count()
    
    def info(self) -> None:
        print(self.chipo.info)
        pass
    
    def num_column(self) -> int:
        return len(self.chipo.columns)
    
    def print_columns(self) -> None:
        print(self.chipo.columns)
        pass

    def most_ordered_item_common(self, order_id_agg):
        most_ordered_item_details = self.chipo \
            .groupby(['item_name']) \
            .agg({'quantity': 'sum', 'order_id': order_id_agg}) \
            .sort_values('quantity', ascending=False) \
            .head(1)
        item_name = most_ordered_item_details.index[0]
        order_id = most_ordered_item_details['order_id'][0]
        quantity = most_ordered_item_details['quantity'][0]
        return item_name, order_id, quantity
    
    def most_ordered_item(self):
        return self.most_ordered_item_common(sum)

    def most_ordered_item_alternative(self):
        return self.most_ordered_item_common(list)

    def total_item_orders(self) -> int:
       return self.chipo.quantity.sum()
   
    def total_sales(self) -> float:
        # 1. Create a lambda function to change all item prices to float.
        item_prices = self.chipo['item_price'].apply(lambda x: float(x[1:]))
        # 2. Calculate total sales.
        return (item_prices * self.chipo.quantity).sum()

    def total_sales_alternative(self) -> float:
        # 1. Create a lambda function to change all item prices to float.
        # self.chipo['item_price'] = self.chipo['item_price'].apply(lambda x: float(x[1:]))
        # 2. Calculate total sales.
        return self.chipo.item_price.sum()

    def num_orders(self) -> int:
        return len(self.chipo.order_id.unique())
    
    def average_sales_amount_per_order(self) -> float:
        return round(self.total_sales() / self.num_orders(), 2)

    def average_sales_amount_per_order_alternative(self) -> float:
        self.chipo['item_price'] = self.chipo['item_price'].apply(lambda x: float(x[1:]))
        return round(self.chipo.groupby(['order_id']).agg({'item_price': sum}).mean()[0], 2)

    def num_different_items_sold(self) -> int:
        return len(self.chipo.item_name.unique())
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # 1. convert the dictionary to a DataFrame
        # https://stackoverflow.com/a/62031043/6073927
        df = pd.DataFrame.from_records(list(letter_counter.items()), columns=['item_name', 'count'])
        # 2. sort the values from the top to the least value and slice the first 5 items
        df = df.sort_values('count', ascending=False).head(x)
        # 3. create a 'bar' plot from the DataFrame
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html
        bar_plot = df.plot.bar(x='item_name', y='count', rot=0)
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        bar_plot.set_xlabel("Items")
        bar_plot.set_ylabel("Number of Orders")
        bar_plot.set_title("Most popular items")
        # 5. show the plot. Hint: plt.show(block=True).
        plt.show(block=True)
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # 1. create a list of prices by removing dollar sign and trailing space.
        self.chipo['item_price'] = self.chipo['item_price'].apply(lambda x: float(x[1:]))
        # 2. groupby the orders and sum it.
        df = self.chipo.groupby(['order_id']).agg(order_price=('item_price', sum), num_items=('quantity', sum))
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        scatter_plot = df.plot.scatter(x='order_price', y='num_items', s=50, c='blue', rot=0)
        # 4. set the title and labels.
        #       title: Number of items per order price
        #       x: Order Price
        #       y: Num Items
        scatter_plot.set_title("Number of items per order price")
        scatter_plot.set_xlabel("Order Price")
        scatter_plot.set_ylabel("Num Items")
        plt.show(block=True)
        pass
    

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926
    # assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()


def test_alternative() -> None:
    solution = Solution()
    item_name, order_id, quantity = solution.most_ordered_item_alternative()
    assert item_name == 'Chicken Bowl'
    assert len(order_id) == 726
    assert quantity == 761
    assert 18.81 == solution.average_sales_amount_per_order_alternative()
    assert 34500.16 == solution.total_sales_alternative()


if __name__ == "__main__":
    # execute only if run as a script
    test()
    # test_alternative()
