## Social Network Ads - KNN Classification

This project demonstrates the implementation of the K-Nearest Neighbors (KNN) classification algorithm using a dataset of social network ads. The aim is to predict whether a user will purchase a product based on their age and estimated salary.

### Dataset

The dataset used in this project is `Social_Network_Ads.csv` which contains the following columns:
- `User ID`: Unique identifier for each user.
- `Gender`: Gender of the user.
- `Age`: Age of the user.
- `EstimatedSalary`: Estimated salary of the user.
- `Purchased`: Whether the user purchased the product (1) or not (0).

### Requirements

To run this project, you'll need the following Python libraries:
- pandas
- scikit-learn
- matplotlib

You can install them using pip:
```bash
pip install pandas scikit-learn matplotlib
```

### Code Explanation

1. **Importing Libraries:**
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.metrics import accuracy_score
   import matplotlib.pyplot as plt
   ```

2. **Loading the Dataset:**
   ```python
   df = pd.read_csv("Social_Network_Ads.csv")
   x1 = df["Age"]
   x2 = df["EstimatedSalary"]
   label = df["Purchased"]
   features = list(zip(x1, x2))
   ```

3. **Splitting the Dataset:**
   ```python
   x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.20)
   ```

4. **Training the Model:**
   ```python
   mymodel = KNeighborsClassifier(n_neighbors=3)
   mymodel.fit(x_train, y_train)
   ```

5. **Making Predictions:**
   ```python
   y_predict = mymodel.predict(x_test)
   ```

6. **Evaluating the Model:**
   ```python
   accuracy = accuracy_score(y_test, y_predict)
   print("\nACCURACY = ")
   print(accuracy)
   ```

7. **Visualizing the Data:**
   - **Histogram of Purchased:**
     ```python
     plt.hist(df['Purchased'], bins=10)
     plt.grid(True)
     plt.show()
     ```

   - **Scatter Plot of Age vs Estimated Salary:**
     ```python
     plt.figure(figsize=(8, 6))
     plt.scatter(df['Age'], df['EstimatedSalary'], color='purple', alpha=0.5)
     plt.title('Scatter Plot of Age vs Estimated Salary')
     plt.xlabel('Age')
     plt.grid(True)
     plt.show()
     ```

### Results

The accuracy of the KNN classifier is printed to the console. Additionally, a histogram showing the distribution of the `Purchased` feature and a scatter plot of `Age` vs `EstimatedSalary` are displayed.

### Conclusion

This project provides a basic example of how to use the KNN algorithm for classification tasks. The visualizations help in understanding the distribution and relationship between the features in the dataset.

### Acknowledgements

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

Feel free to contribute, open issues, and provide suggestions for improving the project. Happy coding!
