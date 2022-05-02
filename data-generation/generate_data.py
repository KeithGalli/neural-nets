import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_data(samples):
	points = []
	for i in range(samples):
		x = np.random.uniform(-5, 5)
		y = np.random.uniform(-5, 5)

		if -1 < x < 0:
			if y > math.sin(x) and y < x**2*math.sin(x):
				label = 0
			elif y < math.tan(x):
				label = 0
			elif y > math.log(x+7):
				label = 0
			else:
				label = 1

		if 0 < x < 1:
			if y < math.sin(x) and y > x**2*math.sin(x):
				label = 0
			elif y > math.tan(x) and y < math.log(x+7):
				label = 0
			else:
				label = 1

		if 1 < x < 2.858:
			if y > math.sin(x) and y < math.log(x+7):
				label = 0
			elif y > math.sin(x) and y > x**2*math.sin(x):
				label = 0
			elif y < math.tan(x):
				label = 0
			else:
				label = 1

		if -1 > x > -math.pi:
			if y < math.sin(x) and y > math.sin(x)*x**2:
				label=0
			elif y > math.tan(x) and y < math.log(x+7):
				label = 0
			else:
				label = 1

		if x > 2.858:
			if y > math.sin(x)*x**2 and y > math.tan(x) and y < math.log(x+7):
				label = 1
			else: 
				label = 0

		if x < -math.pi:
			if y > math.sin(x) and y < math.log(x+7) and y < math.sin(x)*x**2:
				label = 1
			elif y < math.tan(x):
				label = 1
			elif y > math.log(x+7) and y < math.sin(x):
				label = 1
			else:
				label = 0

		points.append([x,y,label])

	df = pd.DataFrame(np.array(points), columns=['x','y','color'])

	return df

# color_dict = {0: 'red', 1: 'blue', 2: 'green'}

# raw_df = df.copy()
# raw_df.to_csv('example.csv', index=False)