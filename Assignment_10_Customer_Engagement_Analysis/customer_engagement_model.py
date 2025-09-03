import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
import numpy as np

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("Assignment_10/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")
print(df.shape)
print(df.head())

# ----------------------------
# Rename columns to remove spaces for statsmodels formulas
# ----------------------------
df.columns = [col.replace(" ", "_") for col in df.columns]
print("Renamed columns:", df.columns.tolist())

# ----------------------------
# Create Engaged column
# ----------------------------
df['Engaged'] = df['Response'].apply(lambda x: 0 if x == 'No' else 1)
print(df.head())

# ----------------------------
# Engagement rate
# ----------------------------
engagement_rate_df = pd.DataFrame(
    df.groupby('Engaged').count()['Response'] / df.shape[0] * 100.0
)
print(engagement_rate_df)
print(engagement_rate_df.T)

# ----------------------------
# Engagement by Offer Type
# ----------------------------
engagement_by_offer_type_df = pd.pivot_table(
    df, values='Response', index='Renew_Offer_Type', columns='Engaged', aggfunc=len
).fillna(0.0)
engagement_by_offer_type_df.columns = ['Not Engaged', 'Engaged']
print(engagement_by_offer_type_df)

engagement_by_offer_type_df.plot(
    kind='pie',
    figsize=(15, 7),
    startangle=90,
    subplots=True,
    autopct=lambda x: '%0.1f%%' % x
)
plt.show()

# ----------------------------
# Engagement by Sales Channel
# ----------------------------
engagement_by_sales_channel_df = pd.pivot_table(
    df, values='Response', index='Sales_Channel', columns='Engaged', aggfunc=len
).fillna(0.0)
engagement_by_sales_channel_df.columns = ['Not Engaged', 'Engaged']
print(engagement_by_sales_channel_df)

engagement_by_sales_channel_df.plot(
    kind='pie',
    figsize=(15, 7),
    startangle=90,
    subplots=True,
    autopct=lambda x: '%0.1f%%' % x
)
plt.show()

# ----------------------------
# Boxplots of Total Claim Amount
# ----------------------------
ax = df[['Engaged', 'Total_Claim_Amount']].boxplot(
    by='Engaged', showfliers=False, figsize=(7,5)
)
ax.set_xlabel('Engaged')
ax.set_ylabel('Total Claim Amount')
ax.set_title('Total Claim Amount Distributions by Engagement')
plt.suptitle("")
plt.show()

ax = df[['Engaged', 'Total_Claim_Amount']].boxplot(
    by='Engaged', showfliers=True, figsize=(7,5)
)
ax.set_xlabel('Engaged')
ax.set_ylabel('Total Claim Amount')
ax.set_title('Total Claim Amount Distributions by Engagement (with outliers)')
plt.suptitle("")
plt.show()

# ----------------------------
# Factorize categorical variables
# ----------------------------
# Gender
gender_values, gender_labels = df['Gender'].factorize()
df['GenderFactorized'] = gender_values

# Education
categories = pd.Categorical(
    df['Education'],
    categories=['High School or Below', 'Bachelor', 'College', 'Master', 'Doctor']
)
df['EducationFactorized'] = categories.codes

# ----------------------------
# Logistic regression: categorical variables
# ----------------------------
formula_categorical = "Engaged ~ GenderFactorized + EducationFactorized"
logit_model_cat = sm.logit(formula=formula_categorical, data=df)
logit_fit_cat = logit_model_cat.fit()
print(logit_fit_cat.summary())

# ----------------------------
# Logistic regression: continuous variables
# ----------------------------
continuous_vars = [
    'Customer_Lifetime_Value', 'Income', 'Monthly_Premium_Auto',
    'Months_Since_Last_Claim', 'Months_Since_Policy_Inception',
    'Number_of_Open_Complaints', 'Number_of_Policies',
    'Total_Claim_Amount'
]

formula_continuous = "Engaged ~ " + " + ".join(continuous_vars)
logit_model_cont = sm.logit(formula=formula_continuous, data=df)
logit_fit_cont = logit_model_cont.fit()
print(logit_fit_cont.summary())
