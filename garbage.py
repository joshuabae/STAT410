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


# It seems because of the large difference in scales and units, standardizing is a better transformation for this data set.

# lines plots dont seem terribly useful in showing confidence intervals and what not (too much fluctuation)
"""
columns = df.drop(columns=['Life_Expectancy']).columns
for column, i in zip(columns,range(len(columns))):
    plt.subplot(5,4,i+1)
    sns.lineplot(df[column], df['Life_Expectancy'])  
    plt.tight_layout()
    sns.set(rc={'figure.figsize':(20,20)})

plt.savefig('lineplot.png')
"""


# If False, perform Welch’s t-test, which does not assume equal population variance

def calculate_t_stat(df):
    t_values = list()
    p_values = list()
    for variable in df.drop(columns=['Life_Expectancy']).columns:
        # with equal_var set to false, Welch’s t-test is used, which does not assume equal population variance
        statistic, pval = stats.ttest_ind(df[variable], df['Life_Expectancy'], equal_var=False)
        t_values.append((variable, abs(statistic)))
        p_values.append((variable, pval))

    t_values = sorted(t_values, reverse=True, key=lambda x: x[1])
    p_values = sorted(p_values, key=lambda x: x[1])

    print('Magnitude of t-statistics for each variable:')
    for value in t_values:
        print(value)

    print('\n')

    print('P-value for each variable:')
    for value in p_values:
        print(value)

    return None


# The greater the magnitude of T (smaller the p-value), the greater the evidence against the null hypothesis
calculate_t_stat(df)

"""
P-value for each variable:
('thinness_1to19_years', 1.2632538534674278e-129)
('thinness_5to9_years', 1.3041808717888162e-129)
('GDP', 5.567972134066403e-109)
('Percentage_Exp', 3.2765243232164745e-90)
('Alcohol', 1.2650993766501536e-82)
('HIV/AIDS', 3.314497828780362e-77)
('Adult_Mortality', 4.580315136952981e-69)
('Under_Five_Deaths', 1.5701198328417583e-48)
('Infant_Deaths', 1.547737989687624e-47)
('Tot_Exp', 8.435358140054619e-44)
('HepatitisB', 5.35847678501304e-34)
('Diphtheria', 2.2490595619042558e-33)
('Polio', 1.1149387992302962e-31)
('Measles', 3.614802713212176e-31)
"""

# some columns have very skewed data (with very many large values)
# skewed_variables = ['Under_Five_Deaths', 'Infant_Deaths', 'Tot_Exp', 'Population', 'Measles', 'HIV/AIDS', 'GDP']
# for variable in skewed_variables:
#     df[variable] = np.log(df[variable])