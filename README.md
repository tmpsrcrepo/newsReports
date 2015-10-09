# newsReports
News reports mining

Preprocessing steps:
  1. Extract news report contents from LexisNexis(preprocessing_p1.py)
  2. Applied Name Entity Recognition on the content (Java)
  3. Designed an algorithm to extract street coordinates from news contents on GDelt data sets:
          use a rule-based algorithm + regular expression to find the address -> concatenate with city,state,country name provided by GDelt -> get geocoded. 

  4. Use machine learning algorithms to look for addresses -> use GBM on bigrams of scraped web contents



  
  

