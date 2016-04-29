#!/usr/bin/python

import datetime
import pandas as pd
import matplotlib.pyplot as plt

SUBMISSION_DATE = datetime.datetime(2016, 9, 16, 10, 00)
START_WRITING_DATE = datetime.datetime(2016, 8, 16, 10, 00)

now = datetime.datetime.now()

time_to_submission = SUBMISSION_DATE - now
time_to_writing = START_WRITING_DATE - now

print ("Days to Submission : ",SUBMISSION_DATE - now)
print ("Days to Writing : ",START_WRITING_DATE - now)



data = pd.read_csv('time-data.csv', header=0)
data = data.fillna(0)
print(data.head())

ax = data.plot( x='Date', y=['Ideal.1', 'Actual.1'],
                grid=True)
plt.show()
