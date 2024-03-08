# Predicting \[feature\] in Spotify Data

Ethan Cohen

## Approach

In order to predict [feature], I started by using a simple linear regressor.

## Quirks

Unlike your standard linear regression model, the gradient is divided by its L2 norm (when larger than 1) to return a unit vector. Initial attempts at getting the model up and running saw the gradient oscillate around the zero vector with its norm increasing with each step.

Regularization is necessary for this to work.