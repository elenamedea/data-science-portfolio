# Data Science portfolio

## Artificial Neural Networks


<div align="justify">The goal of this project is to focus on building an Artificial Neural Network recognizing objects on images made by the webcam. 

Throughout this project I implemented a Feed-Forward Neural Network and Backpropagation from Scratch on make moons dataset, a Convolutional Neural Network with Keras on MNIST dataset and finally I classified the images made with webcam with Pre-trained Networks(VGG16, MobileNet2).

### To Do:
- Update and clean the notebooks 
- Explore further the hyperparameters of the networks
 </div><br>



## Markov Chain-Monte Carlo (MCMC) Simulation


<div align="justify">In this project, I teamed up with my colleague Moritz von Ketelhodt to write an program simulating and predicting customer behaviour between departments/aisles in a supermarket, applying Markov Chain modeling and Monte-Carlo simulation.  

The project involved the following tasks:

### Data Wrangling 
See the notebook/[supermarket_data_wrangling.ipynb](https://github.com/elenamedea/data-science-portfolio/blob/main/MCMC_simulation/supermarket_data_wrangling.ipynb).

### Data Analysis and Expolration 
See the notebook/[supermarket_EDA.ipynb](https://github.com/elenamedea/data-science-portfolio/blob/main/MCMC_simulation/supermarket_EDA.ipynb).

### Calculating Transition Probabilities between the aisles  (5x5 crosstab)
See the notebook/[customer_transition_matrix.ipynb](https://github.com/elenamedea/data-science-portfolio/blob/main/MCMC_simulation/customer_transition_matrix.ipynb).

### Creating a Customer Class
See the notebook/[customer_class.ipynb](https://github.com/elenamedea/data-science-portfolio/blob/main/MCMC_simulation/customer_class.ipynb).

### Running MCMC (Markov-Chain Monte-Carlo) simulation for a single class customer 
See the simulation/[customer_class_one_customer_ simulation_ES.py](https://github.com/elenamedea/data-science-portfolio/blob/main/MCMC_simulation/customer_class_one_customer_simulation_ES.ipynb).

### Extending the simulation to multiple customers
See the simulation/[one_script.py](https://github.com/elenamedea/data-science-portfolio/blob/main/MCMC_simulation/one_script.ipynb).

### To Do:
- Visualization of the supermarket layout and the simulation of the customer behaviour based on the transition probabilities
- Displaying the avatars at the exit location
- Displaing path of the avatars' move between the locations 
</div><br>



## Supervised Machine Learning: Classification - Kaggle's Titanic Challenge


<div align="justify"> This project approaches a classic Machine Learning problem, with a classication model to predict the survival of Titanic passenger based on the features in the dataset of Kaggle's  Titanic - Machine Learning from Disaster.</div><br> 

<div align="justify">Based on the Exploratory Data Analysis (plotted missing values and the correlation between survival and the different data categories) selected the most significant features and dropped the ones which cannot contribute to accurate prediction.</div><br> 

<div align="justify">The data was trained on Scikit-learn's LogisticRegression and RandomForestClassifier models.</div><br>

Data source: [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview).<br>



## Supervised Machine Learning: Regression - Bicycle Rental Forecast


<div align="justify">The goal of this project is to build a regression model, in order to predict the total number of rented bycicles in each hour based on time and weather features, optimizing the accuracy of the model for RMSLE, using Kaggle's "Bike Sharing Demand" dataset that provides hourly rental data spanning two years.</div><br> 

<div align="justify">After extracting datetime features, highly correlated variables were dropped via feature selection (correlation analysis, Variance Inflation Factor) to avoid multicollienarity. I compared more linear regression models with one another (PossionRegressor, PolinomialFeatures, Lasso, Ridge, RandomForestRegressor) based on R2 and RMSLE scores. </div><br>

Data source: [Kaggle: Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data).<br>



## Natural Language Processing (NLP): Text Classification


<div align="justify">The main goal of this project was to build a text classification model on song lyrics to predict the artist from a piece of text, additionally, to make user-inputs ((artists, lyrics) possible in CLI.</div><br> 

<div align="justify">Through web scraping with BeautifulSoup, the song-lyrics of selected artists are extracted from lyrics.com. I built two functions on how to handle the scraping data (extract the song lyrics directly from htmls OR download and save the song lyrics locally as .txt files from every single song lyrics url). In any case, all lyrics will be loaded from the .txt files to create corpus. </div><br>

<div align="justify">In the model pipeline, Tfidfvectorizer (TF-IDF) transforms the words of the corpus into a matrix, count-vectorizes and normalizes them at once by default. For classification, the multinomial Naive Bayes classifier MultinomialNB() was used which is suitable for classification with discrete features like word counts for text classification. 

### To Do:
- Text pre-processing, word-tokenizer and word-lemmatizer of Natural Language Toolkit (NLTK) in order to "clean" the extracted texts
- Debug CLI
 </div><br>



## Time Series Analysis: Temperature Forecast


<div align="justify">For this project, I applied the ARIMA model for a short-term temperature forecast. After visualizing the trend, the seasonality and the remainder of the time series data,  I inspected the lags of the Autocorrelation (ACF) and Partial Auto Correlation Functions (PACF) plots to determine the parameters of the ARIMA odel (p, d, q) and run tests such as ADF and KPSS for checking stationarity (time dependence).</div><br> 

Data source: [European Climate Assessment Dataset](https://www.ecad.eu).<br>
<br>



## Unsupervised learning: Recommender Systems


<p style='text-align: justify;'>This project refers to a movie recommender built with a web interface. The movie recommender is based on the NMF approach, and creates predictions for movies from their ratings average to recommend movies that would most likely be appreciated by that new similar user. 
Trained on 'small' dataset of <a href="https://grouplens.org/datasets/movielens/" target="_blank">MovieLens</a>. 

### To Do:
- Finish and clean the code for the flask app
- Use Streamlit to re-create the app
</p>


**All projects were developed under the scope of Data Science Bootcamp of Spiced Academy.
