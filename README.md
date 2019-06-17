# grab-safety
Based on telematics data, how might we detect if the driver is driving dangerously?

Telematics data are collected from phone’s built in accelerometer and gyroscope.

First step of the design is reading the dataset as provided, preprocessing the data by filtering the outlier based on gps accuracy.

Next, preprocessing also includes offeseting the acceleration signal and generate magnitude of signals in order to have phone orientation independent. 
Time series bearing changes by time is also taking into consideration. 

After these initial steps, random forest and Bayesian classifier are used to identify safe or dangerous driving.
