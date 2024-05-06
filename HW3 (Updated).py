# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:52:37 2024

@author: arifm
"""

# PPHA 30537
# Spring 2024
# Homework 3

# Mahnoor Arif
# Mahnoor Arif
#coding988


#NOTE: All of the plots the questions ask for should be saved and committed to
# your repo under the name "q1_1_plot.png" (for 1.1), "q1_2_plot.png" (for 1.2),
# etc. using fig.savefig. If a question calls for more than one plot, name them
# "q1_1a_plot.png", "q1_1b_plot.png",  etc.

# Question 1.1: With the x and y values below, create a plot using only Matplotlib.
# You should plot y1 as a scatter plot and y2 as a line, using different colors
# and a legend.  You can name the data simply "y1" and "y2".  Make sure the
# axis tick labels are legible.  Add a title that reads "HW3 Q1.1".

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

x = pd.date_range(start='1990/1/1', end='1991/12/1', freq='MS')
y1 = np.random.normal(10, 2, len(x))
y2 = [np.sin(v)+10 for v in range(len(x))]

# Question 1.1
plt.figure(figsize=(10, 6))
plt.scatter(x, y1, label='y1', color='orange')
plt.plot(x, y2, label='y2', color='purple')
plt.title('HW3 Q1.1')
plt.xlabel('Date')
plt.ylabel('Results')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig('q1_1_plot.png')

# Question 1.2: Using only Matplotlib, reproduce the figure in this repo named
# question_2_figure.png.

plt.figure(figsize=(8, 6))
plt.plot(x, y1, label='y1', color='orange')
plt.plot(x, y2, label='y2', color='green')
plt.title('HW3 Q1.2')
plt.xlabel('Date')
plt.ylabel('Results')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig('question_2_figure.png')


# Question 1.3: Load the mpg.csv file that is in this repo, and create a
# plot that tests the following hypothesis: a car with an engine that has
# a higher displacement (i.e. is bigger) will get worse gas mileage than
# one that has a smaller displacement.  Test the same hypothesis for mpg
# against horsepower and weight.

import pandas as pd
import matplotlib.pyplot as plt


# Loading the dataset
mpg_data = pd.read_csv(r'C:\Users\arifm\OneDrive\Documents\GitHub\HW3\mpg.csv')

# Plotting the figure
plt.figure(figsize=(18, 6))

# Plot 1: mpg versus displacement
plt.subplot(1, 3, 1)
plt.scatter(mpg_data['displacement'], mpg_data['mpg'], alpha=0.7)
plt.title('MPG & Displacement')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.savefig('question_1_3a_figure.png')
# Plot 2: mpg versus horsepower
plt.subplot(1, 3, 2)
plt.scatter(mpg_data['horsepower'], mpg_data['mpg'], alpha=0.7)
plt.title('MPG & Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.savefig('question_1_3b_figure.png')
# Plot 3: mpg versus weight
plt.subplot(1, 3, 3)
plt.scatter(mpg_data['weight'], mpg_data['mpg'], alpha=0.7)
plt.title('MPG & Weight')
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.tight_layout()
plt.show()
plt.savefig('question_1_3c_figure.png')

# Question 1.4: Continuing with the data from question 1.3, create a scatter plot 
# with mpg on the y-axis and cylinders on the x-axis.  Explain what is wrong 
# with this plot with a 1-2 line comment.  Now create a box plot using Seaborn
# that uses cylinders as the groupings on the x-axis, and mpg as the values
# up the y-axis.

import seaborn as sns
# Plotting mpg versus cylinders using scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(mpg_data['cylinders'], mpg_data['mpg'])
plt.title('MPG & Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('MPG')
plt.show()
plt.savefig('question_1_4a_figure.png')
#The plot lacks any clarity to show any trend or pattern which makes it challenging to differentiate between individual data points ##

# Plotting box plot using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x='cylinders', y='mpg', data=mpg_data)
plt.title('MPG Distribution of the Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('MPG')
plt.show()
plt.savefig('question_1_4b_figure.png')



# Question 1.5: Continuing with the data from question 1.3, create a two-by-two 
# grid of subplots, where each one has mpg on the y-axis and one of 
# displacement, horsepower, weight, and acceleration on the x-axis.  To clean 
# up this plot:
#   - Remove the y-axis tick labels (the values) on the right two subplots - 
#     the scale of the ticks will already be aligned because the mpg values 
#     are the same in all axis.  
#   - Add a title to the figure (not the subplots) that reads "Changes in MPG"
#   - Add a y-label to the figure (not the subplots) that says "mpg"
#   - Add an x-label to each subplot for the x values
# Finally, use the savefig method to save this figure to your repo.  If any
# labels or values overlap other chart elements, go back and adjust spacing.

# Create the two-by-two grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Plotting each subplot
for i, column in enumerate(['displacement', 'horsepower', 'weight', 'acceleration']):
    row_index = i // 2
    col_index = i % 2
    ax = axes[row_index, col_index]
    ax.scatter(mpg_data[column], mpg_data['mpg'])
    ax.set_title(f'MPG vs. {column.capitalize()}')
    ax.set_xlabel(column.capitalize())
    ax.set_ylabel('MPG')

# Remove y-axis tick labels on the right two subplots
for ax in axes[:, 1]:
    ax.yaxis.set_ticklabels([])

# Add a title to the figure
fig.suptitle('Changes in MPG', fontsize=18)

# Add a y-label to the figure
fig.text(0.06, 0.5, 'mpg', ha='center', va='center', rotation='vertical')

# Adjust spacing
plt.tight_layout()

# Save the figure
plt.savefig('question_1_5_figure.png')

# Show the plot
plt.show()


# Question 1.6: Are cars from the USA, Japan, or Europe the least fuel
# efficient, on average?  Answer this with a plot and a one-line comment.

# Plotting box plot using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x='origin', y='mpg', data=mpg_data)
plt.title('Fuel Efficiency Across Countries')
plt.xlabel('Region')
plt.ylabel('MPG')
plt.show()
##The cars in the US are least fuel efficient compared to Japan and Europe

plt.savefig('question_1_6_figure.png')

# Question 1.7: Using Seaborn, create a scatter plot of mpg versus displacement,
# while showing dots as different colors depending on the country of origin.
# Explain in a one-line comment what this plot says about the results of 
# question 1.6.

# Plotting scatter plot using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='displacement', y='mpg', hue='origin', data=mpg_data)
plt.title('MPG versus Displacement by Country of Origin')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.legend(title='Origin')
plt.show()
#The cars of the US show a greater degree of variance in displacement, compared to cars of japan or europe origin
plt.savefig('question_1_7_figure.png')


# Question 2: The file unemp.csv contains the monthly seasonally-adjusted unemployment
# rates for US states from January 2020 to December 2022. Load it as a dataframe, as well
# as the data from the policy_uncertainty.xlsx file from homework 2 (you do not have to make
# any of the changes to this data that were part of HW2, unless you need to in order to 
# answer the following questions).

# 2.1: Loading and Merging both dataframes together
unemp_data = pd.read_csv(r'C:\Users\arifm\OneDrive\Documents\GitHub\HW3\unemp.csv')
epu_data = pd.read_excel(r'C:\Users\arifm\OneDrive\Documents\GitHub\HW3\policy_uncertainty.xlsx')
unemp_data.columns = unemp_data.columns.str.lower()  #converting column names to lowercase for dataframe unemployment
epu_data.columns = epu_data.columns.str.lower()  #converting column names to lowercase for dataframe policy uncertainty

#defining a function to convert state names to abbreviations
import us
def state_abbrev(state_fullname):
    try:
        state = us.states.lookup(state_fullname)
        if state:
            return state.abbr
        else:
            return None
    except ValueError:
        return None
epu_data['state'] = epu_data['state'].apply(state_abbrev)

# converting from month year to date format of epu dataframe and converting to same date format
epu_data['date'] = pd.to_datetime(epu_data[['year', 'month']].assign(day=1)) 

epu_data['date'] = epu_data['date'].dt.strftime('%Y-%m-%d') 

# Merging dataframes "Unemployment" & "Policy Uncertainty"
data_ump_epu = pd.merge(unemp_data, epu_data, on='state', how='inner')
data_ump_epu.dropna(inplace=True)
print(data_ump_epu.columns)

#    2.2: Calculate the log-first-difference (LFD) of the EPU-C data
import pandas as pd
import statsmodels.api as sm
data_ump_epu['LFD_EPU_C'] = data_ump_epu['epu_composite'].diff().apply(lambda x: x / data_ump_epu['epu_composite'].shift(1))
data_ump_epu.dropna(inplace=True)


#    2.2: Select five states and create one Matplotlib figure that shows the unemployment rate
#         and the LFD of EPU-C over time for each state. Save the figure and commit it with 
#         your code.
five_states = ['Lousiana', 'Maine', 'Kentucky', 'Florida', 'Illinois']

emp_lfd_state_data = data_ump_epu[data_ump_epu['state'].isin(five_states)]
emp_lfd_state_data.dropna(subset=['unemp_rate', 'LFD_EPU_C'], inplace=True)##dropping missing values

# Filter data for the five states
emp_lfd_state_data = data_ump_epu[data_ump_epu['state'].isin(five_states)]

# Drop missing values from the selected columns
emp_lfd_state_data.dropna(subset=['unemp_rate', 'LFD_EPU_C'], inplace=True)


print(emp_lfd_state_data.info())
print(emp_lfd_state_data.columns)

##using a lopp to creating a plot
plt.figure(figsize=(12, 8))
for state in five_states:
    state_subset = data_ump_epu[data_ump_epu['state'] == state]
    plt.plot(state_subset['date_y'], state_subset['unemp_rate'], label=f'{state} Unemployment Rate')
    plt.plot(state_subset['date_y'], state_subset['LFD_EPU_C'], label=f'{state} LFD EPU-C')
plt.title('Unemployment Rate and LFD of EPU-C Over Time for the five US States')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('question_2_2_figure.png')
plt.show()

#    2.3: Using statsmodels, regress the unemployment rate on the LFD of EPU-C and fixed
#         effects for states. Include an intercept.

X = sm.add_constant(emp_lfd_state_data['LFD_EPU_C']) #adding a intercept
y = emp_lfd_state_data['unemp_rate']
ols_model = sm.OLS(y, X)
reg_results = ols_model.fit()
print(reg_results.summary())

#    2.4: Print the summary of the results, and write a 1-3 line comment explaining the basic
#         interpretation of the results (e.g. coefficient, p-value, r-squared), the way you 
#         might in an abstract.

# Results summary
# Overall there is a negative correlation of unemployment rate with the log-first-difference (LFD) of the EPU-C data; however, the relationship is significant with a p-value of <0.05.
#The R-squared value suggests that a reasonable degree of the variation in the unemployment rate can be explained by the the log-first-difference (LFD) of the EPU-C data


