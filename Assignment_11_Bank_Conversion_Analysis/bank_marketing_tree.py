import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("Assignment_11/bank-full.csv", sep=";")
print(df.shape)
print(df.head())

# ----------------------------
# Create conversion column
# ----------------------------
df['conversion'] = df['y'].apply(lambda x: 0 if x == 'no' else 1)
print(df.head())

# ----------------------------
# Conversion rate
# ----------------------------
conversion_rate_df = pd.DataFrame(df.groupby('conversion').count()['y'] / df.shape[0] * 100.0)
print(conversion_rate_df)
print(conversion_rate_df.T)

# ----------------------------
# Conversion by marital status
# ----------------------------
conversion_rate_by_marital = df.groupby('marital')['conversion'].sum() / df.groupby('marital')['conversion'].count() * 100
ax = conversion_rate_by_marital.plot(
    kind='bar',
    color='skyblue',
    grid=True,
    figsize=(10,7),
    title='Conversion Rates by Marital Status'
)
ax.set_xlabel('Marital Status')
ax.set_ylabel('Conversion rate (%)')
plt.show()

# ----------------------------
# Conversion by job
# ----------------------------
conversion_rate_by_job = df.groupby('job')['conversion'].sum() / df.groupby('job')['conversion'].count() * 100
ax = conversion_rate_by_job.plot(
    kind='barh',
    color='skyblue',
    grid=True,
    figsize=(10,7),
    title='Conversion Rates by Job'
)
ax.set_xlabel('Conversion rate (%)')
ax.set_ylabel('Job')
plt.show()

# ----------------------------
# Default by conversion (pie)
# ----------------------------
default_by_conversion_df = pd.pivot_table(
    df, values='y', index='default', columns='conversion', aggfunc=len
).fillna(0)
default_by_conversion_df.columns = ['non_conversions', 'conversions']
default_by_conversion_df.plot(
    kind='pie',
    figsize=(15,7),
    startangle=90,
    subplots=True,
    autopct=lambda x: '%0.1f%%' % x
)
plt.show()

# ----------------------------
# Boxplots of balance
# ----------------------------
ax = df[['conversion', 'balance']].boxplot(by='conversion', showfliers=True, figsize=(10,7))
ax.set_xlabel('Conversion')
ax.set_ylabel('Average Bank Balance')
ax.set_title('Average Bank Balance Distributions by Conversion')
plt.suptitle("")
plt.show()

ax = df[['conversion', 'balance']].boxplot(by='conversion', showfliers=False, figsize=(10,7))
ax.set_xlabel('Conversion')
ax.set_ylabel('Average Bank Balance')
ax.set_title('Average Bank Balance Distributions by Conversion (no outliers)')
plt.suptitle("")
plt.show()

# ----------------------------
# Conversion by number of contacts
# ----------------------------
conversions_by_num_contacts = df.groupby('campaign')['conversion'].sum() / df.groupby('campaign')['conversion'].count() * 100
ax = conversions_by_num_contacts.plot(
    kind='bar',
    figsize=(10,7),
    title='Conversion Rates by Number of Contacts',
    grid=True,
    color='skyblue'
)
ax.set_xlabel('Number of Contacts')
ax.set_ylabel('Conversion Rate (%)')
plt.show()

# ----------------------------
# Encode categorical variables
# ----------------------------
# Month to numeric
months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
df['month'] = df['month'].apply(lambda x: months.index(x)+1)

# Jobs encoding
jobs_encoded_df = pd.get_dummies(df['job'], prefix='job')
df = pd.concat([df, jobs_encoded_df], axis=1)

# Marital encoding
marital_encoded_df = pd.get_dummies(df['marital'], prefix='marital')
df = pd.concat([df, marital_encoded_df], axis=1)

# Housing and loan binary
df['housing'] = df['housing'].apply(lambda x: 1 if x=='yes' else 0)
df['loan'] = df['loan'].apply(lambda x: 1 if x=='yes' else 0)

# ----------------------------
# Define features and response
# ----------------------------
features = ['age', 'balance', 'campaign', 'previous', 'housing'] + \
           list(jobs_encoded_df.columns) + list(marital_encoded_df.columns)
response_var = 'conversion'

# ----------------------------
# Train Decision Tree
# ----------------------------
dt_model = tree.DecisionTreeClassifier(max_depth=5)
dt_model.fit(df[features], df[response_var])
print("Classes:", dt_model.classes_)

# ----------------------------
# Visualize Decision Tree with matplotlib
# ----------------------------
plt.figure(figsize=(20,10))
tree.plot_tree(
    dt_model,
    feature_names=features,
    class_names=['0','1'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()
