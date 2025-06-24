import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

df = pd.read_csv("Titanic-Dataset_Cleaned.csv")

summary_stats = df.describe()

numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numeric_cols].hist(bins=20, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col], color='orange')
    plt.title(f'Boxplot - {col}')
plt.tight_layout()
plt.show()

sns.pairplot(df[['Survived', 'Age', 'Fare', 'SibSp', 'Parch']], hue='Survived', palette='coolwarm')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

fig = px.histogram(df, x='Pclass', color='Survived', barmode='group', title='Survival by Passenger Class')
fig.show()

fig = px.histogram(df, x='Sex', color='Survived', barmode='group', title='Survival by Sex')
fig.show()

fig = ff.create_distplot(
    [df[df['Survived'] == 1]['Age'], df[df['Survived'] == 0]['Age']],
    group_labels=['Survived', 'Not Survived'],
    show_hist=False,
    colors=['green', 'red']
)
fig.update_layout(title_text='Age Distribution: Survived vs Not Survived')
fig.show()
