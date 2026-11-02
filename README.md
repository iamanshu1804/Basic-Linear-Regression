# Regression Fundamentals with Python

This project demonstrates the end-to-end workflow for two supervised learning
regression problems:

1. **Simple linear regression**: predicting height from weight.
2. **Multiple linear regression**: predicting California district house values
   from several demographic and geographic features.

The notebooks focus on understanding the complete machine-learning workflow:
loading data, visualising relationships, preparing features, splitting data,
training a model, making predictions, and evaluating errors.

## Repository contents

| File | Description |
|---|---|
| [`Linear_Regression.ipynb`](./Linear_Regression.ipynb) | Simple linear regression using the height-weight dataset |
| [`Multiple_Linear_Regression.ipynb`](./Multiple_Linear_Regression.ipynb) | Multiple linear regression using the California Housing dataset |
| [`height-weight.csv`](./height-weight.csv) | Input data for the simple regression notebook |

## 1. Simple linear regression notebook

The [`Linear_Regression.ipynb`](./Linear_Regression.ipynb) notebook models the
relationship between one independent variable and one dependent variable:

```text
Weight -> Height
```

### What the notebook does

1. Imports NumPy, pandas, Matplotlib, and scikit-learn.
2. Loads [`height-weight.csv`](./height-weight.csv).
3. Displays the data and creates a scatter plot of weight against height.
4. Separates the data into:
   - `X`: the input feature (`Weight`)
   - `y`: the target (`Height`)
5. Splits the data into training and testing sets using an 80/20 split.
6. Standardises the weight feature with `StandardScaler`.
7. Trains a `LinearRegression` model on the training data.
8. Displays the learned coefficient and intercept.
9. Draws the fitted regression line for the training and test data.
10. Predicts heights for unseen weights, including an example prediction for a
    person weighing 90 kg.
11. Evaluates the model using:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R-squared
    - Adjusted R-squared
12. Uses residual plots to examine the model's prediction errors.

### What this notebook demonstrates

This notebook is an introduction to the idea that a linear model estimates a
straight-line relationship:

```text
predicted height = intercept + coefficient * weight
```

Because there is only one input feature, the relationship can be visualised
directly on a two-dimensional graph. The notebook also shows why the same
scaler fitted on the training data must be used when preparing new input
values.

## 2. Multiple linear regression notebook

The [`Multiple_Linear_Regression.ipynb`](./Multiple_Linear_Regression.ipynb)
uses scikit-learn's California Housing dataset. It predicts the median house
value of a California district from eight input features:

- `MedInc`: median income
- `HouseAge`: median house age
- `AveRooms`: average number of rooms
- `AveBedrms`: average number of bedrooms
- `Population`: district population
- `AveOccup`: average household occupancy
- `Latitude`
- `Longitude`

The target is named `Price` in the notebook. It is measured in units of
`$100,000`, so a target value of `3.5` represents approximately `$350,000`.

### What the notebook does

1. Loads the California Housing dataset with
   `fetch_california_housing`.
2. Creates a pandas DataFrame containing the eight features.
3. Adds the target values as the `Price` column.
4. Inspects the dataset with:
   - `head()`
   - `info()`
   - `isnull().sum()`
   - `describe()`
5. Calculates a correlation matrix and displays it as a heatmap.
6. Separates the feature matrix `X` from the target `Y`.
7. Splits the data into training and test sets using a fixed random state.
8. Trains a `LinearRegression` model using all eight features.
9. Predicts house values for the test set.
10. Calculates R-squared and adjusted R-squared.
11. Uses three diagnostic graphs:
    - Actual versus predicted values
    - The distribution of residuals
    - Residuals versus predicted values

### Current model result

The saved notebook output reports:

```text
R-squared:          0.5970
Adjusted R-squared: 0.5966
```

This means the model explains approximately 59.7% of the variation in the
test-set house values. The remaining error is expected because house prices
depend on complex geographic and economic relationships that a straight-line
model cannot fully represent.

### How to interpret the visualisations

- **Correlation heatmap**: shows the strength and direction of linear
  relationships between features and the target. `MedInc` has the strongest
  individual positive correlation with `Price`.
- **Actual versus predicted plot**: points closer to the ideal diagonal line
  represent more accurate predictions.
- **Residual distribution**: shows whether errors are centred around zero and
  whether they are skewed or affected by outliers.
- **Residuals versus predicted plot**: a random cloud around zero is desirable.
  Curves, funnels, or clusters can indicate non-linearity, changing error
  variance, or groups of observations that the model handles differently.

The correlation heatmap also highlights possible multicollinearity. For
example, `AveRooms` and `AveBedrms` are strongly related, as are `Latitude`
and `Longitude`. This means some features contain overlapping information and
their individual regression coefficients should be interpreted carefully.

## Technologies used

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## How to run the notebooks

1. Clone this repository.
2. Install the required packages:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open either notebook and run the cells from top to bottom.

## Possible future improvements

The current notebooks establish interpretable baselines. Possible next steps
include:

- Add MAE and RMSE to the multiple regression notebook.
- Use five-fold cross-validation instead of relying on one train-test split.
- Compare Linear Regression with Ridge, Lasso, Random Forest, and Gradient
  Boosting models.
- Add a baseline model that always predicts the mean target value.
- Calculate Variance Inflation Factor (VIF) to investigate multicollinearity.
- Add a model-comparison table and document the results.
- Improve the plots with reference lines, units, legends, and explanatory
  captions.
- Discuss the age of the California Housing data and the limitations of using
  it for current real-estate decisions.
