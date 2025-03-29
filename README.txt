Group 8 DDAM Project project structure:
1. Data_Understanding.ipynb - preliminary dataset analysis and EDA with vizualizations
	1.1 auto_eda.py - functions to automatically analyze dataset based on predefined feature types
	1.2 utils.py - functions to handle repeatable tasks like loading dataframe from file with specified format or creating spark session
	1.3 viz.py - functions to vizualize findings from EDA
2. Supervised.ipynb - feature creation, logistic regression, decision tree, random forest
3. Clustering_Task.ipynb - sentence embedding, averaging embeddings by regions and finding the optimal clustering
4. survival_analysis.ipynb - Kaplan Meier curve, AFT Regression Model, number of days to reach top200
	- 4.1. ne_110m_admin_0_countries - directory with .shp files for visualizations
