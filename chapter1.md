---
title       : Chapter 1
description : Stuff and stuff
attachments :

--- type:NormalExercise lang:R xp:100 skills:1 key:ca2686ceeb
## How it works

Welcome to our Kaggle House Prices: Advanced Regression tutorial. In this tutorial, you will explore how to tackle the Kaggle House Prices: Advanced Regression Techniques competition using R and Machine Learning. In case you're new to R, it's recommended that you first take our free <a target="_blank" href="https://www.datacamp.com/courses/intro-to-R-for-data-science"> Introduction to R for Data Science<a/> Tutorial. Furthermore, while not required, familiarity with machine learning techniques is a plus so you can get the maximum out of this tutorial.

In the editor on the right, you should type R code to solve the exercises. When you hit the 'Submit Answer' button, every line of code is interpreted and executed by R and you get a message whether or not your code was correct. The output of your R code is shown in the console in the lower right corner. R makes use of the `#` sign to add comments; these lines are not run as R code, so they will not influence your result.

You can also execute R commands straight in the console. This is a good way to experiment with R code, as your submission is not checked for correctness.

*** =instructions
- In the editor to the right, you see some R code and annotations. This is what a typical exercise will look like.
- To complete the exercise and see how the interactive environment works add the code to compute y and hit the `Submit Answer` button. Don't forget to print the result.

*** =hint
- Just add a line of R code that calculates the product of 6 and 9, just like the example in the sample code!

*** =pre_exercise_code
```{R}
```

*** =sample_code
```{R}
#Compute x = 4 * 3 and print the result
x = 4 * 3
print(x)

#Compute y = 6 * 9 and print the result
y = 6*9; print(y)

```

*** =solution
```{R}
#Compute x = 4 * 3 and print the result
x = 4 * 3
print(x)

#Compute y = 6 * 9 and print the result
y = 6*9; print(y)
```

*** =sct
```{R}


success_msg("Awesome! See how the console shows the result of the R code you submitted? Now that you're familiar with the interface, let's get down to business!")
```

--- type:NormalExercise lang:R xp:100 skills:2 key:672930f088
## Get the data with Pandas 
For many the dream of owning a home doesn't include searching for the perfect basement ceiling height or the proximity to an east-west railroad. However, the 79 explanatory variables describing (almost) every aspect of residential homes used in the Kaggle House Price Competition show that there is much more that influences price negotiations than the number of bedrooms or a white-picket fence.

In this course, you will learn how to apply machine learning techniques to predict the final price of each home using R.

"The potential for creative feature engineering provides a rich opportunity for fun and learning. This dataset lends itself to advanced regression techniques like random forests and gradient boosting with the popular XGBoost library. We encourage Kagglers to create benchmark code and tutorials on Kernels for community learning. Top kernels will be awarded swag prizes at the competition close." 

*** =instructions
- First, import the Pandas library as pd.
- Load the test data similarly to how the train data is loaded.
- Inspect the first couple rows of the loaded dataframes using the .head() method with the code provided.

*** =hint
- You can load in the training set with ```train = pd.read_csv(train_url)```
- To print a variable to the console, use the print function on a new line.

*** =pre_exercise_code
```{R}

```

*** =sample_code
```{R}
# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/train.csv"
train = read.csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/test.csv"
test = read.csv(test_url)

#Print the `head` of the train and test dataframes
print(head(train))
print(head(test))

```

*** =solution
```{R}
# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/train.csv"
train = read.csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/test.csv"
test = read.csv(test_url)

#Print the `head` of the train and test dataframes
print(head(train))
print(head(test))

```

*** =sct
```{R}
#msg = "Have you correctly imported the `pandas` package? Use the alias `pd`."
#test_import("pandas",  not_imported_msg = msg,  incorrect_as_msg = msg)

#msg = "Do not touch the code that specifies the URLs of the training and test set csvs."
#test_object("train_url", undefined_msg = msg, incorrect_msg = msg)
#test_object("test_url", undefined_msg = msg, incorrect_msg = msg)

#msg = "Make sure you are using the `read_csv()` function correctly"
#test_function("pandas.read_csv", 1,
#              args=None,
#              not_called_msg = msg,
#              incorrect_msg = msg,)

#test_function("pandas.read_csv", 2,
#              args=None,
#             not_called_msg = msg,
#              incorrect_msg = msg)

#msg = "Don't forget to print the first few rows of `train` with the `.head()` method"
#test_function("print", 1, not_called_msg = msg, incorrect_msg = msg)

#msg = "Don't forget to print the first few rows of `test` with the `.head()` method"
#test_function("print", 2, not_called_msg = msg, incorrect_msg = msg)

success_msg("Well done! Now that your data is loaded in, let's see if you can understand it.")
```

--- type:MultipleChoiceExercise lang:R xp:50 skills:2 key:5e47ef16d2
## Understanding your data 

Before starting with the actual analysis, it's important to understand the structure of your data. The variables loaded in the previous exercise, train and test, are data frames, R's way of representing a dataset. You can easily explore a data frame using the function `str()`. `str()` gives you information such as the data types in the data frame (e.g. int for integer), the number of observations, and the number of variables. It is a great way to get a feel for the contents of the data frame.

The 2 data frames are already loaded into your workspace. Apply `str()` to each variable to see its dimensions and basic composition. Which of the following statements is correct?

*** =instructions
- The training set has 1460 observations and 81 variables, count for LotFrontage is 1233.
- The training set has 1459 observations and 80 variables, count for LotFrontage is 1459.
- The testing set has 1459 observations and 81 variables, count for LotFrontage is 1234.
- The testing set has 1459 observations and 80 variables, count for LotFrontage is 1232.

*** =hint
- hint

*** =pre_exercise_code
```{R}
train = read.csv("http://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/train.csv")
test = read.csv("http://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/test.csv")
```

*** =sct

```{R}

msg1 = "Incorrect. Maybe have a look at the hint."
msg2 = "Wrong, try again. Maybe have a look at the hint."
msg3 = "Not so good... Maybe have a look at the hint."
msg4 = "Great job!"
test_mc(correct = 4, msgs = [msg1, msg2, msg3, msg4])

success_msg("Well done! Now move on and explore some of the features in more detail.")

```
--- type:NormalExercise xp:100 skills:1 key:f21d7e0656
## Explore and Visualize

Another great way to explore your data is to create a few visualizations. This can help you better understand the structure and potential limitations of particular variables. 

Check out the structure of the variables in `train` with `str()`. You will see the majority of them are categorical. If there aren't too many categories in a variable, a bar chart can be a great way to visualize and digest your data. 

CHANGE THIS
The two variables show a pattern but the thrid doesn't look like there is much of a pattern to the roof style and sale price
descrieb the variables
- description 
You can see descriptions of all of the variables on the competition page [here](https://www.drivendata.org/competitions/7/page/25/). 

*** =instructions
- The code given uses the package `ggplot2` to create a bar chart for the variable `quantity` using the aesthetic `fill` to partition by `status_group`
- Using similar syntax, make a similar plot for `quality_group` 
- Then again for `waterpoint_type`

*** =hint
Use the same code that is provided for the `quantity` plot. Simply change the first argument in the command.

*** =pre_exercise_code
```{r,eval=FALSE}
train = read.csv("http://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/train.csv")
```

*** =sample_code
```{r,eval=FALSE}
# Load the ggplot package and examine train
library(ggplot2)
str(train)

# Create bar plot for Garage Type
ggplot(train, aes(x = SalePrice/1000, fill = GarageType)) + geom_histogram() + 
  theme(legend.position = "top")

# Create bar plot for Kitchen Quality
ggplot(___, aes(x = SalePrice/1000, fill = KitchenQual)) + geom_histogram() + 
  theme(legend.position = "top")

# Create bar plot for Roof Style
ggplot(___, aes(x = SalePrice/1000, fill = RoofStyle)) + geom_histogram() + 
  theme(legend.position = "top")

```

*** =solution
```{r,eval=FALSE}
# Load the ggplot package and examine train
library(ggplot2)
str(train)

# Create bar plot for Garage Type
ggplot(train, aes(x = SalePrice/1000, fill = GarageType)) + geom_histogram() + 
  theme(legend.position = "top")

# Create bar plot for Kitchen Quality
ggplot(train, aes(x = SalePrice/1000, fill = KitchenQual)) + geom_histogram() + 
  theme(legend.position = "top")

# Create bar plot for Roof Style
ggplot(train, aes(x = SalePrice/1000, fill = RoofStyle)) + geom_histogram() + 
  theme(legend.position = "top")


```

*** =sct
```{r,eval=FALSE}
msg <- "There is no need to change the commands in the sample code."
test_function_v2("ggplot", "x", eval = FALSE, index = 1, 
                 incorrect_msg = msg)
test_function_v2("ggplot", "x", eval = FALSE, index = 2, 
                 incorrect_msg = paste(msg, " Simply change the `x` value in `qplot()` to `quality_group`"))
test_function_v2("ggplot", "x", eval = FALSE, index = 3, 
                 incorrect_msg = paste(msg, " Simply change the `x` value in `qplot()` to `waterpoint_type`"))
test_error()
success_msg("Awesome! Now let's look at a few more visualizations.")
```

--- type:NormalExercise lang:R xp:100 skills:1 key:1eeaaeb294
## Square Feet vs Lot Size  

The object of the Kaggle competition is to is to predict the sale price of the properties listed in the test data set. Let's look at the training data set and see what we can gather from the data at a quick glace. We can see the sale price of the propeties listed in the training data set by using the standard bracket notation to select a single column of a DataFrame:

`train["SalePrice"]`

We can also look at some of the variabels to try to find obvious patterns. Let's start with the size of the house represented by `

*** =instructions
- Calculate and print 

*** =hint
- hint

*** =pre_exercise_code
```{R}
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/train.csv")
test = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/production/course_2470/datasets/test.csv")
```

*** =sample_code
```{R}

# Sale Prices
print()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model


# Fit the model
model = ols("z ~ x + y", data).fit()

# Print the summary
print(model.summary())

print("\nRetrieving manually the parameter estimates:")
print(model._results.params)
# should be array([-4.99754526,  3.00250049, -0.50514907])

# Peform analysis of variance on fitted linear model
anova_results = anova_lm(model)

print('\nANOVA results')
print(anova_results)

plt.show()





# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

```

*** =solution
```{R}

# Sale Prices
print(train["SalePrice"])

```

*** =sct

```{R}
msg = "Make sure you are using the bracket method correctly."
test_function("print", 1,
              not_called_msg= msg,
              incorrect_msg = msg)

success_msg("Well done!.")

```

--- type:NormalExercise lang:R xp:100 skills:2 key:b8f71cf4de
## Does age play a role?

Another variable that could influence survival is age; since it's probable that children were saved first. You can test this by creating a new column with a categorical variable `Child`. `Child` will take the value 1 in cases where age is less than 18, and a value of 0 in cases where age is greater than or equal to 18. 

To add this new variable you need to do two things (i) create a new column, and (ii) provide the values for each observation (i.e., row) based on the age of the passenger.

Adding a new column with Pandas in R is easy and can be done via the following syntax:

```
your_data["new_var"] = 0
```

This code would create a new column in the `train` DataFrame titled `new_var` with `0` for each observation.

To set the values based on the age of the passenger, you make use of a boolean test inside the square bracket operator. With the `[]`-operator you create a subset of rows and assign a value to a certain variable of that subset of observations. For example,

```
train["new_var"][train["Fare"] > 10] = 1
```

would give a value of `1` to the variable `new_var` for the subset of passengers whose fares greater than 10. Remember that `new_var` has a value of `0` for all other values (including missing values).

A new column called `Child` in the `train` data frame has been created for you that takes the value `NaN` for all observations.

*** =instructions
- Set the values of `Child` to `1` is the passenger's age is less than 18 years. 
- Then assign the value `0` to observations where the passenger is greater than or equal to 18 years in the new `Child` column. 
- Compare the normalized survival rates for those who are <18 and those who are older. Use code similar to what you had in the previous exercise.

*** =hint
Suppose you wanted to add a new column `clothes` to the `test` set, then give all males the value `"pants"` and the others `"skirt"`:

```
test["clothes"] = "skirt"

test["clothes"][test["Sex"] == "male"] = "pants"
```

*** =pre_exercise_code

```{R}
import pandas as pd
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
```

*** =sample_code

```{R}
# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.




# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older


```

*** =solution

```{R}
# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
print(train["Child"])

# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

```

*** =sct
```{R}
msg = "Remember to print the new column `Child`. It should be equal to 1 when the passenger's age is under 18 and 0 if the passenger's age is 18 or greater."
test_function("print", 2,
              not_called_msg = msg,
              incorrect_msg = msg)

msg = "Compute the survival proportions for those OVER 18. Refer to the code provided for passengers under 18."
test_function("print", 3,
              not_called_msg = msg,
              incorrect_msg = msg)

success_msg("Well done! As you can see from the survival proportions, age does certainly seem to play a role.")
```

--- type:NormalExercise lang:R xp:100 skills:2 key:f02305d182
## First Prediction

In one of the previous exercises you discovered that in your training set, females had over a 50% chance of surviving and males had less than a 50% chance of surviving. Hence, you could use this information for your first prediction: all females in the test set survive and all males in the test set die. 

You use your test set for validating your predictions. You might have seen that contrary to the training set, the test set has no `Survived` column. You add such a column using your predicted values. Next, when uploading your results, Kaggle will use this variable (= your predictions) to score your performance. 

*** =instructions
- Create a variable `test_one`, identical to dataset `test`
- Add an additional column, `Survived`, that you initialize to zero.
- Use vector subsetting like in the previous exercise to set the value of `Survived` to 1 for observations whose `Sex` equals `"female"`.
- Print the `Survived` column of predictions from the `test_one` dataset.

*** =hint
- To create a new variable, `y`, that is a copy of `x`, you can use `y = x`.
- To initialize a new column `a` in a dataframe `data` to zero, you can use `data['a'] = 0`.
- Have another look at the previous exercise if you're struggling with the third instruction.

*** =pre_exercise_code

```{R}
import pandas as pd
train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")
```

*** =sample_code

```{R}
# Create a copy of test: test_one


# Initialize a Survived column to 0


# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
```

*** =solution

```{R}
# Create a copy of test: test_one
test_one = test

# Initialize a Survived column to 0
test_one["Survived"] = 0

# Set Survived to 1 if Sex equals "female"
test_one["Survived"][test_one["Sex"] == "female"] = 1
print(test_one.Survived)
```

*** =sct

```{R}

test_function("print",
              not_called_msg = "Make sure to define the column `Survived` inside `test_one`",
              incorrect_msg = "Make sure you are assigning 1 to female and 0 to male passengers")

success_msg("Well done! If you want, you can already submit these first predictions to Kaggle [by uploading this csv file](http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/ch1_ex4_solution/my_solution.csv). In the next chapter, you will learn how to make more advanced predictions and create your own .csv file from R.")
```
