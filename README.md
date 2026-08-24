# EDA-project-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

n = 10000

categories = ["Electronics", "Clothing", "Home", "Beauty", "Sports"]
cities = ["Indore", "Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad", "Chennai"]
payment_methods = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]
customer_types = ["New", "Returning", "VIP"]

products = {
    "Electronics": ["Laptop", "Smartphone", "Headphones", "Smartwatch"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Shoes"],
    "Home": ["Chair", "Table", "Lamp", "Bedsheet"],
    "Beauty": ["Face Wash", "Perfume", "Shampoo", "Moisturizer"],
    "Sports": ["Cricket Bat", "Football", "Running Shoes", "Yoga Mat"]
}

dates = pd.date_range("2025-01-01", "2026-06-30", freq="D")

df = pd.DataFrame({
    "Order_ID": np.arange(100001, 100001 + n),
    "Order_Date": np.random.choice(dates, n),
    "Category": np.random.choice(categories, n),
    "City": np.random.choice(cities, n),
    "Payment_Method": np.random.choice(payment_methods, n),
    "Customer_Type": np.random.choice(customer_types, n, p=[0.45, 0.4, 0.15]),
    "Quantity": np.random.randint(1, 6, n),
    "Discount": np.round(np.random.uniform(0, 0.30, n), 2),
    "Rating": np.round(np.random.uniform(2.5, 5, n), 1)
})

df["Product"] = df["Category"].apply(
    lambda x: np.random.choice(products[x])
)

base_price = {
    "Electronics": 30000,
    "Clothing": 1800,
    "Home": 2500,
    "Beauty": 1200,
    "Sports": 2200
}

df["Unit_Price"] = df["Category"].map(base_price)

df["Unit_Price"] = (
    df["Unit_Price"] *
    np.random.uniform(0.7, 1.3, n)
).round(2)

df["Gross_Sales"] = (
    df["Quantity"] * df["Unit_Price"]
).round(2)

df["Discount_Amount"] = (
    df["Gross_Sales"] * df["Discount"]
).round(2)

df["Net_Sales"] = (
    df["Gross_Sales"] - df["Discount_Amount"]
).round(2)

df["Profit_Margin"] = np.random.uniform(0.08, 0.30, n)

df["Profit"] = (
    df["Net_Sales"] * df["Profit_Margin"]
).round(2)

df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

df["Year"] = df["Order_Date"].dt.year

df["Month_Name"] = df["Order_Date"].dt.month_name()

missing_index = np.random.choice(df.index, 100, replace=False)

df.loc[missing_index[:50], "Rating"] = np.nan
df.loc[missing_index[50:], "Payment_Method"] = np.nan

df = pd.concat([df, df.iloc[:20]], ignore_index=True)

print("\nDataset Shape Before Cleaning:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

df["Rating"] = df["Rating"].fillna(df["Rating"].median())

df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")

df.reset_index(drop=True, inplace=True)

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nBasic Statistics:")
print(df.describe())

print("\nTotal Revenue:", round(df["Net_Sales"].sum(), 2))
print("Total Profit:", round(df["Profit"].sum(), 2))
print("Total Orders:", df["Order_ID"].nunique())
print("Average Order Value:", round(df["Net_Sales"].mean(), 2))

category_sales = (
    df.groupby("Category")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales By Category:")
print(category_sales)

category_profit = (
    df.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\nProfit By Category:")
print(category_profit)

city_sales = (
    df.groupby("City")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales By City:")
print(city_sales)

product_sales = (
    df.groupby("Product")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop Products:")
print(product_sales.head(10))

customer_analysis = (
    df.groupby("Customer_Type")
    .agg(
        Orders=("Order_ID", "count"),
        Revenue=("Net_Sales", "sum"),
        Profit=("Profit", "sum"),
        Avg_Rating=("Rating", "mean")
    )
    .sort_values("Revenue", ascending=False)
)

print("\nCustomer Analysis:")
print(customer_analysis)

monthly_sales = (
    df.groupby("Month")["Net_Sales"]
    .sum()
)

monthly_profit = (
    df.groupby("Month")["Profit"]
    .sum()
)

payment_analysis = (
    df.groupby("Payment_Method")["Net_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nPayment Method Analysis:")
print(payment_analysis)

discount_analysis = (
    df.groupby("Category")["Discount"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Discount By Category:")
print(discount_analysis)

correlation = df[
    [
        "Quantity",
        "Discount",
        "Rating",
        "Unit_Price",
        "Gross_Sales",
        "Net_Sales",
        "Profit"
    ]
].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=category_sales.index,
    y=category_sales.values
)

plt.title("Revenue By Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=city_sales.index,
    y=city_sales.values
)

plt.title("Revenue By City")
plt.xlabel("City")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=60)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_profit.index,
    monthly_profit.values,
    marker="o"
)

plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit")
plt.xticks(rotation=60)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Category",
    y="Net_Sales"
)

plt.title("Sales Distribution By Category")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Net_Sales"],
    bins=50,
    kde=True
)

plt.title("Distribution of Order Revenue")
plt.xlabel("Net Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(2000),
    x="Discount",
    y="Net_Sales",
    hue="Category",
    alpha=0.7
)

plt.title("Discount vs Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=payment_analysis.index,
    y=payment_analysis.values
)

plt.title("Revenue By Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=customer_analysis.index,
    y=customer_analysis["Revenue"].values
)

plt.title("Revenue By Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

top_products = product_sales.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_products.values,
    y=top_products.index
)

plt.title("Top 10 Products By Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

category_profit_margin = (
    df.groupby("Category")["Profit"]
    .sum()
    /
    df.groupby("Category")["Net_Sales"].sum()
    * 100
).sort_values(ascending=False)

print("\nProfit Margin By Category:")
print(category_profit_margin.round(2))

best_category = category_sales.idxmax()
best_city = city_sales.idxmax()
best_product = product_sales.idxmax()
best_customer = customer_analysis["Revenue"].idxmax()

print("\n========== BUSINESS INSIGHTS ==========")

print("Best Category:", best_category)
print("Best City:", best_city)
print("Best Product:", best_product)
print("Highest Revenue Customer Type:", best_customer)

print(
    "Highest Revenue Category Sales:",
    round(category_sales.max(), 2)
)

print(
    "Highest Revenue City Sales:",
    round(city_sales.max(), 2)
)

print(
    "Overall Profit Margin:",
    round(
        df["Profit"].sum() /
        df["Net_Sales"].sum() * 100,
        2
    ),
    "%"
)

print(
    "Average Customer Rating:",
    round(df["Rating"].mean(), 2)
)

print("\nProject Completed Successfully.")
