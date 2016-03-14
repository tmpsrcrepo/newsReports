# newsReports
News reports mining

Preprocessing steps:
  1. Extract news report contents from LexisNexis(preprocessing_p1.py)
  2. Applied Name Entity Recognition on the content (Java)
  3. Designed an algorithm to extract street coordinates from news contents on GDelt data sets:
          use a rule-based algorithm + regular expression to find the address -> concatenate with city,state,country name provided by GDelt -> get geocoded. 

Data: WashingtonPosts (2015-2016, ~1000 docs)

Modeling:
  1. Address labeling -> 'WP_adddress.csv'
  2. Expand the context window by k (need to define k at the beginning of the script) -> 'labeling_expand_window.py'
  3. Extract features and label the data (preparing the data frame for modeling) -> 'features_and_labeling.py'
  4. Use machine learning algorithms to look for addresses -> 'model.py'






  
  

