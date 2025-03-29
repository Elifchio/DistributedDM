This repository contains effects of collaborative work of:

Adam Dohojda
Wiktoria Stęczna
Elif Yilmaz
The analysis takes TOP 200 podcast charts from September 2024 to until the middle of October 2024 from various regions on the world and performs several analyses:

what are the semantical commonalities between regions based on podcast descriptions?
can we predict whetever the podcast will go up or down in the charts?
what is the estimated time and what are the factors that contribute to podcast drop off TOP 200?
The project was developed using PySpark library.

Project structure:

1. Data_Understanding.ipynb - preliminary dataset analysis and EDA with vizualizations
	1.1 auto_eda.py - functions to automatically analyze dataset based on predefined feature types
	1.2 utils.py - functions to handle repeatable tasks like loading dataframe from file with specified format or creating spark session
	1.3 viz.py - functions to vizualize findings from EDA
2. Supervised.ipynb - feature creation, logistic regression, decision tree, random forest
3. Clustering_Task.ipynb - sentence embedding, averaging embeddings by regions and finding the optimal clustering
4. survival_analysis.ipynb - Kaplan Meier curve, AFT Regression Model, number of days to reach top200
	- 4.1. ne_110m_admin_0_countries - directory with .shp files for visualizations
5. Report_ddm.pdf - Short report that summarizes the work and the outcomes.

Dataset:
https://www.kaggle.com/datasets/daniilmiheev/top-spotify-podcasts-daily-updated - 229k of records from all over the world about the ranking and change of rankings of podcasts
