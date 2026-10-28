import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a = pd.read_csv("superstore.csv")

print("========== DATASET PREVIEW ==========")
print(a.head())

print("\n========== LAST 5 ROWS ==========")
print(a.tail())

print("\n========== SHAPE ==========")
print(a.shape)

print("\n========== COLUMNS ==========")
print(a.columns)

print("\n========== DATA TYPES ==========")
print(a.dtypes)

print("\n========== INFO ==========")
print(a.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(a.describe())

print("\n========== CATEGORICAL SUMMARY ==========")
print(a.describe(include="object"))

print("\n========== MISSING VALUES ==========")
print(a.isnull().sum())

print("\n========== MISSING VALUE PERCENTAGE ==========")
print((a.isnull().sum() / len(a)) * 100)

print("\n========== DUPLICATES ==========")
print(a.duplicated().sum())

a = a.drop_duplicates()

print("\n========== DUPLICATES AFTER CLEANING ==========")
print(a.duplicated().sum())

a["Order_Date"] = pd.to_datetime(a["Order_Date"], errors="coerce")
a["Ship_Date"] = pd.to_datetime(a["Ship_Date"], errors="coerce")

print("\n========== DATE TYPES ==========")
print(a[["Order_Date", "Ship_Date"]].dtypes)

a["Ship_Days"] = (
    a["Ship_Date"] - a["Order_Date"]
).dt.days

a["Order_Year"] = a["Order_Date"].dt.year
a["Order_Month"] = a["Order_Date"].dt.month
a["Order_Month_Name"] = a["Order_Date"].dt.strftime("%B")
a["Order_Quarter"] = a["Order_Date"].dt.quarter
a["Order_Day"] = a["Order_Date"].dt.day
a["Order_Day_Name"] = a["Order_Date"].dt.strftime("%A")

print("\n========== NEW COLUMNS ==========")
print(a.head())

print("\n========== UNIQUE VALUES ==========")
print("Orders:", a["Order_ID"].nunique())
print("Customers:", a["Customer_ID"].nunique())
print("Products:", a["Product_ID"].nunique())
print("Categories:", a["Category"].nunique())
print("Sub Categories:", a["Sub_Category"].nunique())
print("States:", a["State"].nunique())
print("Cities:", a["City"].nunique())

print("\n========== VALUE COUNTS ==========")

print("\nShip Mode")
print(a["Ship_Mode"].value_counts())

print("\nSegment")
print(a["Segment"].value_counts())

print("\nRegion")
print(a["Region"].value_counts())

print("\nCategory")
print(a["Category"].value_counts())

print("\nSub Category")
print(a["Sub_Category"].value_counts())

print("\n========== TOTAL SALES ==========")
total_sales = a["Sales"].sum()
print(total_sales)

print("\n========== TOTAL PROFIT ==========")
total_profit = a["Profit"].sum()
print(total_profit)

print("\n========== TOTAL QUANTITY ==========")
total_quantity = a["Quantity"].sum()
print(total_quantity)

print("\n========== TOTAL DISCOUNT ==========")
total_discount = a["Discount"].sum()
print(total_discount)

print("\n========== AVERAGE SALES ==========")
print(a["Sales"].mean())

print("\n========== AVERAGE PROFIT ==========")
print(a["Profit"].mean())

print("\n========== AVERAGE DISCOUNT ==========")
print(a["Discount"].mean())

print("\n========== AVERAGE QUANTITY ==========")
print(a["Quantity"].mean())

print("\n========== PROFIT MARGIN ==========")
profit_margin = (total_profit / total_sales) * 100
print(profit_margin)

print("\n========== SALES BY CATEGORY ==========")
category_sales = (
    a.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(category_sales)

print("\n========== PROFIT BY CATEGORY ==========")
category_profit = (
    a.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(category_profit)

print("\n========== QUANTITY BY CATEGORY ==========")
category_quantity = (
    a.groupby("Category")["Quantity"]
    .sum()
    .sort_values(ascending=False)
)
print(category_quantity)

print("\n========== SALES BY SUB CATEGORY ==========")
subcategory_sales = (
    a.groupby("Sub_Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(subcategory_sales)

print("\n========== PROFIT BY SUB CATEGORY ==========")
subcategory_profit = (
    a.groupby("Sub_Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(subcategory_profit)

print("\n========== TOP 10 PRODUCTS BY SALES ==========")
top_products_sales = (
    a.groupby("Product_Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_products_sales)

print("\n========== TOP 10 PRODUCTS BY PROFIT ==========")
top_products_profit = (
    a.groupby("Product_Name")["Profit"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_products_profit)

print("\n========== WORST 10 PRODUCTS BY PROFIT ==========")
worst_products_profit = (
    a.groupby("Product_Name")["Profit"]
    .sum()
    .sort_values()
    .head(10)
)
print(worst_products_profit)

print("\n========== SALES BY REGION ==========")
region_sales = (
    a.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(region_sales)

print("\n========== PROFIT BY REGION ==========")
region_profit = (
    a.groupby("Region")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(region_profit)

print("\n========== SALES BY STATE ==========")
state_sales = (
    a.groupby("State")["Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(state_sales.head(15))

print("\n========== PROFIT BY STATE ==========")
state_profit = (
    a.groupby("State")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(state_profit.head(15))

print("\n========== WORST STATES BY PROFIT ==========")
print(state_profit.sort_values().head(15))

print("\n========== TOP CITIES BY SALES ==========")
city_sales = (
    a.groupby("City")["Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(city_sales.head(15))

print("\n========== TOP CUSTOMERS ==========")
customer_sales = (
    a.groupby(["Customer_ID", "Customer_Name"])["Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(customer_sales.head(15))

print("\n========== CUSTOMER PROFIT ==========")
customer_profit = (
    a.groupby(["Customer_ID", "Customer_Name"])["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(customer_profit.head(15))

print("\n========== SALES BY SEGMENT ==========")
segment_sales = (
    a.groupby("Segment")["Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(segment_sales)

print("\n========== PROFIT BY SEGMENT ==========")
segment_profit = (
    a.groupby("Segment")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(segment_profit)

print("\n========== SALES BY SHIP MODE ==========")
ship_sales = (
    a.groupby("Ship_Mode")["Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(ship_sales)

print("\n========== PROFIT BY SHIP MODE ==========")
ship_profit = (
    a.groupby("Ship_Mode")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(ship_profit)

print("\n========== MONTHLY SALES ==========")
monthly_sales = (
    a.groupby(["Order_Year", "Order_Month"])["Sales"]
    .sum()
)
print(monthly_sales)

print("\n========== MONTHLY PROFIT ==========")
monthly_profit = (
    a.groupby(["Order_Year", "Order_Month"])["Profit"]
    .sum()
)
print(monthly_profit)

print("\n========== YEARLY SALES ==========")
yearly_sales = (
    a.groupby("Order_Year")["Sales"]
    .sum()
)
print(yearly_sales)

print("\n========== YEARLY PROFIT ==========")
yearly_profit = (
    a.groupby("Order_Year")["Profit"]
    .sum()
)
print(yearly_profit)

print("\n========== QUARTERLY SALES ==========")
quarterly_sales = (
    a.groupby(["Order_Year", "Order_Quarter"])["Sales"]
    .sum()
)
print(quarterly_sales)

print("\n========== SALES BY DAY ==========")
day_sales = (
    a.groupby("Order_Day_Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(day_sales)

print("\n========== DISCOUNT ANALYSIS ==========")
discount_profit = (
    a.groupby("Discount")["Profit"]
    .mean()
    .sort_values()
)
print(discount_profit)

print("\n========== DISCOUNT VS SALES ==========")
discount_sales = (
    a.groupby("Discount")["Sales"]
    .mean()
)
print(discount_sales)

print("\n========== NEGATIVE PROFIT ORDERS ==========")
loss_orders = a[a["Profit"] < 0]

print(loss_orders.head())
print("Number of loss rows:", len(loss_orders))

print("\n========== LOSS BY CATEGORY ==========")
loss_category = (
    loss_orders.groupby("Category")["Profit"]
    .sum()
    .sort_values()
)
print(loss_category)

print("\n========== LOSS BY SUB CATEGORY ==========")
loss_subcategory = (
    loss_orders.groupby("Sub_Category")["Profit"]
    .sum()
    .sort_values()
)
print(loss_subcategory)

print("\n========== OUTLIERS IN SALES ==========")
Q1 = a["Sales"].quantile(0.25)
Q3 = a["Sales"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

sales_outliers = a[
    (a["Sales"] < lower) |
    (a["Sales"] > upper)
]

print("Lower:", lower)
print("Upper:", upper)
print("Sales outliers:", len(sales_outliers))

print("\n========== OUTLIERS IN PROFIT ==========")
Q1 = a["Profit"].quantile(0.25)
Q3 = a["Profit"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

profit_outliers = a[
    (a["Profit"] < lower) |
    (a["Profit"] > upper)
]

print("Lower:", lower)
print("Upper:", upper)
print("Profit outliers:", len(profit_outliers))

print("\n========== CORRELATION ==========")

numeric_columns = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Ship_Days"
]

correlation = a[numeric_columns].corr()

print(correlation)

plt.figure(figsize=(10,7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))

sns.histplot(
    a["Sales"],
    kde=True
)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.show()

plt.figure(figsize=(8,5))

sns.histplot(
    a["Profit"],
    kde=True
)

plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    x=a["Sales"]
)

plt.title("Sales Boxplot")
plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    x=a["Profit"]
)

plt.title("Profit Boxplot")
plt.show()

plt.figure(figsize=(8,5))

sns.barplot(
    x=category_sales.index,
    y=category_sales.values
)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))

sns.barplot(
    x=category_profit.index,
    y=category_profit.values
)

plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_products_sales.values,
    y=top_products_sales.index
)

plt.title("Top 10 Products by Sales")
plt.xlabel("Sales")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))

sns.barplot(
    x=worst_products_profit.values,
    y=worst_products_profit.index
)

plt.title("Worst 10 Products by Profit")
plt.xlabel("Profit")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))

sns.barplot(
    x=region_sales.index,
    y=region_sales.values
)

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.show()

plt.figure(figsize=(8,5))

sns.barplot(
    x=segment_sales.index,
    y=segment_sales.values
)

plt.title("Sales by Customer Segment")
plt.xlabel("Segment")
plt.ylabel("Sales")
plt.show()

plt.figure(figsize=(12,6))

plt.plot(
    yearly_sales.index,
    yearly_sales.values,
    marker="o"
)

plt.title("Yearly Sales Trend")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.grid()
plt.show()

monthly = (
    a.groupby(a["Order_Date"].dt.to_period("M"))["Sales"]
    .sum()
)

plt.figure(figsize=(14,6))

plt.plot(
    monthly.index.astype(str),
    monthly.values,
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=a,
    x="Discount",
    y="Profit"
)

plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=a,
    x="Sales",
    y="Profit"
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    data=a,
    x="Category",
    y="Profit"
)

plt.title("Profit Distribution by Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    data=a,
    x="Segment",
    y="Profit"
)

plt.title("Profit Distribution by Segment")
plt.show()

print("\n========== FINAL BUSINESS INSIGHTS ==========")

print("Best Category by Sales:", category_sales.idxmax())
print("Best Category by Profit:", category_profit.idxmax())
print("Best Region by Sales:", region_sales.idxmax())
print("Best Region by Profit:", region_profit.idxmax())
print("Best Segment by Sales:", segment_sales.idxmax())
print("Best Ship Mode by Sales:", ship_sales.idxmax())
print("Top Product by Sales:", top_products_sales.idxmax())
print("Top Product by Profit:", top_products_profit.idxmax())
print("Worst Product by Profit:", worst_products_profit.idxmin())
print("Best State by Sales:", state_sales.idxmax())
print("Best State by Profit:", state_profit.idxmax())
print("Worst State by Profit:", state_profit.idxmin())
print("Best Month:", monthly.idxmax())
print("Worst Month:", monthly.idxmin())
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
import pandas as pd

a=pd.read_csv("healthcare_dataset.csv")
# a.head(10)
# a.shape
# a.columns
# a.dtypes
# a.isnull().sum()
# a.duplicated()
# a.describe()
# a["Test Results"].value_counts()
_______________________________________________________________________________________

# a.copy()
# a.isnull().sum()
# a.fillna(a.select_dtypes("number").median())
# a.fillna(a.select_dtypes(include=["object","category"]).mode().iloc[0])
# a.drop_duplicates(inplace=True)
# a.isnull().sum()
# a.duplicated()
# a["Age"].value_counts()
# a["Age"].min()
# a["Age"].max()
# a["Age"].mean()
# a["Billing Amount"].median()
________________________________________________________________________________________
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(8,5))
# plt.plot(a["Medical Condition"])
# plt.show()
# plt.figure(figsize=(8,5))
# plt.plot(a["Gender"])
# plt.show()
# plt.figure(figsize=(8,5))
# plt.plot(a["Age"])
# plt.show()
# plt.figure(figsize=(8,5))
# sns.countplot(a["Medical Condition"])
# plt.show()
# plt.scatter(a["Age"],a["Billing Amount"])
# plt.show()
# sns.heatmap(data=a.select_dtypes(include="number").corr(),annot=True,cmap="coolwarm")
# plt.show()
# sns.boxplot(data=a,y="Billing Amount",color="skyblue")
# plt.show()

_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
# Data Preproccess Practise
import pandas as pd

Data = pd.read_csv("healthcare_dataset.csv")
Data.shape
Data.columns
Data.dtypes
Data.isnull().sum()
Data.duplicated().sum()
Data.describe()
Data["Test Results"].value_counts()
______________________________________________
print(Data.isnull().mean())
print(Data.nunique().value_counts())
for i in Data.select_dtypes(include=["object"]):
    print(i)
    print(Data[i].unique())

Q1 = Data.select_dtypes('number').quantile(0.25)
Q3 = Data.select_dtypes('number').quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outlier = Data[(Data.select_dtypes("number")<lower) | (Data.select_dtypes("number")>upper)]
print(outlier)
print(Data["Test Results"].value_counts(normalize=True)*100)

for i in Data.select_dtypes("number").columns:
    Q1,Q3=Data[i].quantile(0.25),Data[i].quantile(0.75)
    IQR = Q3 - Q1
    Data[i]=Data[i].clip(lower=Q1-1.5*IQR,upper=Q3+1.5*IQR)
__________________________________________________________________________________________________
import pandas as pd
import numpy as np
df = pd.DataFrame({
    "Age": [22, 25, np.nan, 30,  np.nan],
    "Salary": [25000, np.nan, 35000, 40000, 30000],
    "City": ["Indore", "Bhopal", np.nan, "Indore", np.nan],
    "Experience": [1, 2, 3, np.nan, 5]
})
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Experience"]=df["Experience"].fillna(df["Experience"].median())
missing_check = df[["Age","Salary","Experience"]].isnull().sum()
if (missing_check==0).all():
    print("yes")
else:
    print("No")
print(df.isnull().sum())
print(df.isnull().sum()/len(df)*100)
print(df.isnull().sum().sum())
print(df.isnull().sum().idxmax())
print(df.isnull().sum().max())
_____________________________________________________________________________
df = pd.DataFrame({
    "City": ["Indore", "Bhopal", np.nan, "Indore", np.nan, "Bhopal"],
    "Gender": ["Male", np.nan, "Female", "Male", "Female", np.nan],
    "Department": ["IT", "HR", "IT", np.nan, "HR", "IT"]
})
print(df.isnull().sum())
col = ["City","Gender","Department"]
df[col]=df[col].fillna(df[col].mode().iloc[0])
print(df[col].isnull().sum())
if df.isnull().sum().sum()==0:
    print("No")
else:
    print("Yss")
print(df)
_____________________________________________________________________________
c = ["City", "Gender", "Department"]
df[c] =df[c].fillna(df[c].mode().iloc[0])
print(df[c].isnull().sum())
print(df.isnull().sum())
if df.isnull().sum().sum()==0:
    print("no")
else:
    print("No")
________________________________________________
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
df = pd.DataFrame({
    "Age": [22, np.nan, 30, 25, np.nan, 40],
    "Salary": [25000, 30000, np.nan, 40000, 35000, np.nan],
    "City": ["Indore", np.nan, "Bhopal", "Indore", "Ujjain", np.nan],
    "Experience": [1, 2, np.nan, 3, 4, 5]
})
print(df.select_dtypes("number"))
print(df.select_dtypes(include=["object","category"]))
numeric = df.select_dtypes("number").columns
imputer = SimpleImputer(strategy="median")
df[numeric]=imputer.fit_transform(df[numeric])
print(df)
category = df.select_dtypes(include=["object","category"]).columns
impu = SimpleImputer(strategy="most_frequent")
df[category]=impu.fit_transform(df[category])
print(df)
if df.isnull().sum().sum()==0:
    print("no")
else:
    print("yss")
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________

from common_imports import *

Data = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

print(Data.shape)
print(Data[:10])
print(Data.columns)
print(Data.isnull().sum())

print(Data.duplicated().sum())
Data.drop_duplicates(inplace=True)
print(Data)
Data["Order Date"] = pd.to_datetime(Data["Order Date"])
Data["Ship Date"] = pd.to_datetime(Data["Ship Date"])
Data.to_csv("Sample - Superstore.csv", index=False)
print(Data[["Order Date", "Ship Date"]].dtypes)
print(Data.select_dtypes("number").describe())

plt.hist(Data["Sales"],bins=30)
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Sales Distribution")
plt.show()

plt.hist(Data["Profit"],bins=30)
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.title("Profit Distribution")
plt.show()

plt.boxplot(Data["Discount"])
plt.xlabel("Descount")
plt.ylabel("Frequency")
plt.title("Discount Distribution")
plt.show()

Data["Category"].value_counts().plot(kind="bar")
plt.xlabel("Category")
plt.ylabel("Frequency")
plt.title("Category Distribution")
plt.show()

Data["Segment"].value_counts().plot(kind="bar")
plt.xlabel("Segment")
plt.ylabel("Count")
plt.title("Segment-wise Count")
plt.show()

Data.groupby("Category")["Profit"].sum().plot(kind="bar")
plt.xlabel("Category")
plt.ylabel("Total Profit")
plt.title("Category-wise Total Profit")
plt.show()

Data.groupby("Region")["Sales"].sum().plot(kind="bar")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.title("Category-wise Total Sales")
plt.show()

plt.scatter(Data["Sales"],Data["Profit"])
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Category- Wise profit")
plt.show()

sns.heatmap(Data[["Sales","Profit","Discount"]].corr(), annot=True)
plt.show()

_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
