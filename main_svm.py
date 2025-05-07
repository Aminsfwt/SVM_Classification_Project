#import the helping function file
from PrepData import *


path = 'heart.csv'
X, y = read_data(path, 'num', 'id')

#split the data into train and test sets
# Same results each time as random_state is set to 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Drop columns with any NaN values in the training set
X_train = X_train.dropna(axis=1)
X_test  = X_test.dropna(axis=1)

y_train = y_train.dropna()
y_test  = y_test.dropna()

#Scale features using StandardScaler
#fit_transform() --> only do it for train data
#transform() --> do it for both train and test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


#Train the SVM model
svm_model = LinearSVC()
svm_model.fit(X_train_scaled, y_train)

#Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Print the first 5 rows of predictions and actual values
print("First 5 predictions:")
print("Y PRED = \n", y_pred[:20])
print("\nFirst 5 actual values:")
print("Y TEST = \n", y_test[:20].values)

# Print the accuracy score
svm_accuracy = accuracy_score(y_test, y_pred)
print(f"\nSVM Model Accuracy: {svm_accuracy:.2f}")

"""
#print coefficients
print("\nCoefficients:")
print(svm_model.coef_)
"""

"""
#confiusion matrix This generates a 2x2 matrix showing the counts of:
True Negatives  (top-left)
False Positives (top-right)
False Negatives (bottom-left)
True Positives  (bottom-right)
"""

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
"""
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Heart Disease', 'Heart Disease'])
disp.plot(cmap=plt.cm.Blues, values_format='.2f')
plt.title('Confusion Matrix')
plt.show()
"""

# Hyperparameter Tuning with GridSearchCV
# Define the parameter grid for hyperparameter tuning
# GridSearchCV will search for the best combination of parameters and evaluate the model using cross-validation
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf'] # Kernel type
}  

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# best parameters
best_params = grid_search.best_params_
print(f"\nBest Parameters: {best_params}")

# The best model found during grid search
best_svm_model = grid_search.best_estimator_
print(f"\nBest SVM Model: {best_svm_model}")

# Evaluate the best model on the test set
y_pred_best = best_svm_model.predict(X_test_scaled)
print('\nBest SVM Model Accuracy: \n', accuracy_score(y_test, y_pred_best))

# For linear kernel, get coefficients from best model
if best_svm_model.kernel == 'linear':
    """
    Extracts the coefficients from a trained SVM model
    coef_[0] returns the weights assigned to each feature in the first class (for binary classification)
    """
    coefficients = best_svm_model.coef_[0]
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": coefficients
    })
    feature_importance.sort_values(by="Coefficient", key=abs, ascending=False, inplace=True)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Coefficient", y="Feature", data=feature_importance)
    plt.title("Feature Importance (SVM Coefficients)")
    plt.show()
else:
    print("Feature importance plot is only available for linear kernel")

"""
#print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
"""



"""
# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the scatter plot using three features
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 4], X.iloc[:, 8], c=y, cmap='viridis')

# Add labels
ax.set_xlabel(X.columns[0])
ax.set_ylabel(X.columns[4])
ax.set_zlabel(X.columns[8])

# Add a colorbar
plt.colorbar(scatter, label='Heart Disease')

plt.title('Heart Disease Data in 3D')
plt.show()
"""
"""feature, ax = plt.subplots(1, 5)
feature.suptitle('Heart Disease Prediction')
for i in range(5):
    ax[i].scatter(X.iloc[:, i], y, color='blue')
    ax[i].set_xlabel(X.columns[i])
    ax[i].set_ylabel(y.name)

plt.tight_layout()
plt.show()"""
