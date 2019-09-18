# StackExchange-Analysis
*--Joel Hensel, Mark Tenzer, Devan Visvalingam, Ada Zhu*

**The goal of this analysis is to answer key questions that would help various StackExchange users interact more positively with the platform.**

To begin with, users in this analysis can be categorized into three roles: 
* Those who are seeking knowledge (the “Questioner”)
* Those who are providing knowledge (the “Answerer”)
* Those governing the platform (the “Moderator”)

The analysis sought answers to the following questions:
* For the Questioner, “will my question get an answer at all?” 
* For the Questioner, “how many answers will my question get?” 
* For the Questioner, “will my question get an accepted answer?” 
* For the Answerer, “how do I have my answer get the highest score?”

*Each of these questions is founded on the notion that if the user knew of certain variables that predicted the outcomes, they could manipulate their posts in order to better achieve their outcomes.*


# Acquiring the Data and Pre-Processing Methods
On StackExchange Data Explorer, we select around ten variables out of over a hundred variables to serve as candidate regressors based on the four business questions we sought to answer. Using both SQL in the acquisition of the data, as well as R in the creation of a dataframe, many of the candidate regressor variables were transformed to facilitate better modeling. Many user attributes, such as the profile picture, were transformed into binary attributes that indicated whether the user has profile picture or not. Several attributes came with the value of “NA” when in reality the value was 0. In order to make these attributes consistently numeric, these NAs were converted to 0s. We also created new variables such as "duration", which captures the time elapsed between when a question is first posted and when the corresponding answer is posted. 

Further transformations were required of certain regressor variables due to the distribution of the raw data. One example is length of the post’s body and also title. The vast majority of posts had body length under 2,500 characters for instance, but some outlier posts had lengths that exceeded 20,000. For both of these regressors, taking the log provided a much more suitable and normal distribution. 
![](images/length_transformation.png)

# As a Questioner, How Many Answers Will My Question Get?

Next, regularization schemes were considered--both lasso and ridge regression.  Possible values for lambda (also known as k) were chosen uniformly along an exponential scale from e^-6 to e^-2 (in steps of e^0.05).  Optimal values were chosen by 10-fold cross validation using cv.glmnet.  In both cases, the optimal value was so small, a positive impact could nearly be ruled out without further analysis.  For lasso, lambda was approximately 0.004; for ridge, 0.010.  Fitting a model with the ridge penalty to the same training set as before, then evaluating out-of-sample (test set) R2 produced nearly the same value: 0.356.  For further analysis, the penalization was ignored due to its ineffectiveness in improving accuracy and its increased model complexity.

Stepwise regression was performed to select the model with the optimal AIC on the training set: there were too many predictors to effectively do exhaustive search on even one train/test split, let alone 100.  The optimal set of predictors was used to fit a Poisson GLM. Performing model selection in the training set did not improve out-of-sample test set performance and the small changes in R^2 ver only 100 splits might have been due to chance. However, when AIC stepwise regression was applied to the full set of observations, the largest variance inflation factors (VIFs) decreased relative to those calculated from a full model of all predictors.  Fewer VIFs exceeded 5, and those that did so barely did.  Therefore, the AIC-selected model was used as the final model.

![](images/vif%20comparison.png)

Decrease in VIFs (light blue) relative to the remaining VIFs (darker blue) after stepwise AIC model selection.  Red columns are VIFs for terms in the full model that were dropped altogether by model selection.  Any increase if VIFs would be shown in grey stacked bars, but none are present.

The coefficients suggested that a questioner who wishes to maximize the number of answers they receive should try to increase a few quantities.  At the level of their user account, they should maximize their account creation date (i.e., have a younger account) and number of views, while taking care to include a location on their profile.  Regarding the question itself, they should maximize its score, number of views, and its number of favorites, while minimizing its length, number of tags, and number of comments.  Although included in the AIC-selected model, the 95% confidence intervals for the coefficients of Profile Image and Website, both indicator variables describing the author’s account profile, contained 0.  These terms were also the only ones insignificant at p < 0.05 on the per-predictor Wald’s t-test.  It was not certain whether they played a positive or negative role.  Intuitively, they should improve the number of answers (due to increased asking-account quality), but it seemed that any effect was very small.

The most significant term, by far, was the view count of the post (| t | = 35).  This was quite intuitive--every answerer must view the post, and the more users who view the post, the more opportunities there are to be answered.  With a Wald t-value about half that of view count, the next-strongest predictor was the post score (| t | = 17), followed by the length of the post body ((| t | = 10).  It is therefore most important to maximize the post views and score while keeping the post itself as brief as possible.  Unfortunately, interpreting the coefficients more directly was made difficult by the extensive transformations performed in the data.  The exponentiation of a Poisson GLM coefficient is the multiplicative factor by which the response increases, given a 1-unit increase in the predictor.  However, multiple predictors were transformed with the reciprocal square root function--though necessary to normalize some predictors with extremely skewed distributions, this transformation obfuscated any interpretability.  The “multiplicative factor if the reciprocal of the square root is increased by 1 unit” would be very difficult to intuitively interpret, but substituting this transformation for a more interpretable one (e.g., log) reduced the model’s performance.


# As an Answerer, How Do I Have My Answer Get the Highest Score?	

We first split the dataset into training and testing with a ratio of 0.8 to 0.2. This split allows us to test our data later using dataset that’s not included when building the model, estimating the accuracy of the model on unseen data in an unbiased manner. We ran a regression on all 9 regressors using the training data. We also applied boxcox function, and the transformation suggested seem to be very similar to our model.   

**Check for Influential Points**

From the Cook’s distance plot, we could see that most of the data points fall between 0 and 0.01, with only a few influential points spreading out. Considering the large size of our dataset, we decided to keep these points in the model to avoid losing any important information.

**Stepwise**

We then ran step regression to check if the model could be improved by model selection. The function stepAIC removed variable **hasProfileImg**, which seemed reasonable since we would assume most of the users that have profile image linked to their account also have “aboutMe” in the profile, and more variability can be explained by “aboutMe”. The VIFs of each of the coefficients in this reduced model are all relatively small, implying that there was no multicollinearity among variables. Our potential model was now: 

log(score) = 2.872e+00 + 2.807e-02 * length + 2.236e-04 * duration   + 3.355e-02 * questionView 
 -1.213e-04 * TitleLen  -8.710e-03 * HasAboutMe + 3.267e-02 * Reputation 
  -2.126e-02 * UserViews  -1.309e-03 * TagCount
  
**Conclusion**  

For answerers to earn the highest scores, not only do they have to provide great answers, but also be selective in which questions they choose to respond to. Unlike questions, longer answers tended to yield positive outcomes for answerers. However, more attributes pertaining to the question itself seemed to influence score. For example, questions with shorter titles, more views, and less tags tended to produce answers with higher scores. Also, user attributes of the answerer, such as Reputation and Views, influenced the score, regardless of its content.

