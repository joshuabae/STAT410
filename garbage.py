# ridge regression has the best performance of all
ridge_model = Ridge()
# lasso says nothing is significant (all weights are 0)
lasso_model = Lasso()
# elastic says only adult_mort, Income_Comp_Of_Resources, Schooling are important (which seems very likely)
elastic_model = ElasticNet()


# regular linear regression stuff
linear_model = LinearRegression()
fitted_linear = linear_model.fit(X_train, y_train)
print(fitted_linear.coef_)
results_linear = fitted_linear.predict(X_test)

fitted_ridge = ridge_model.fit(X_train, y_train)
fitted_lasso = lasso_model.fit(X_train, y_train)
fitted_elastic = elastic_model.fit(X_train, y_train)

results_ridge = fitted_ridge.predict(X_test)
results_lasso = fitted_lasso.predict(X_test)
results_elastic = fitted_elastic.predict(X_test)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(100, 50))
fig.suptitle('Fitted vs. Actual')
ax1.scatter(results_ridge, y_test, s=1000)
ax2.scatter(results_lasso, y_test, s=1000)
ax3.scatter(results_elastic, y_test, s=1000)
ax4.scatter(results_linear, y_test, s=1000)