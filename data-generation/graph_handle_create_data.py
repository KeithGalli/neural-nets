import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from generate_data import create_data

def easy_line_2(samples):
  x1 = np.random.uniform(0,5, samples)
  y1 = x1 + np.random.normal(1,0.25, samples)
  z1 = np.zeros(y1.shape)
  
  red = np.column_stack((x1,y1,z1))
  
  x2 = np.random.uniform(0,5,samples)
  y2 = x2 - np.random.normal(1,0.25, samples)
  z2 = np.ones(y2.shape)
  
  blue = np.column_stack((x2,y2,z2))
  
  final= np.row_stack((red,blue))
  
  return pd.DataFrame(final, columns=['x','y','color'])

def easy_line_3(samples):
  x1 = np.random.uniform(0,5, samples)
  y1 = x1 + np.random.normal(1,0.25, samples)
  z1 = np.zeros(y1.shape)
  
  red = np.column_stack((x1,y1,z1))
  
  x2 = np.random.uniform(0,5,samples)
  y2 = x2 - np.random.normal(1,0.25, samples)
  z2 = np.ones(y2.shape)
  
  blue = np.column_stack((x2,y2,z2))
  
  x3 = np.random.uniform(0,5,samples)
  y3 = x3 - np.random.normal(5,0.25, samples)
  z3 = np.ones(y2.shape)*2
  
  green = np.column_stack((x3,y3,z3))
  
  final= np.row_stack((red,blue, green))
  
  return pd.DataFrame(final, columns=['x','y','color'])
    
def quadratic(samples):
  x1 = np.random.uniform(-5,5, samples)
  y1 = x1**2 + np.random.uniform(1,2, samples)
  z1 = np.zeros(y1.shape)

  red = np.column_stack((x1,y1,z1))

  x2 = np.random.uniform(-5,5,samples)
  y2 = x2**2 - np.random.uniform(1,2, samples)
  z2 = np.ones(y2.shape)

  blue = np.column_stack((x2,y2,z2))

  final = np.row_stack((red,blue))

  return pd.DataFrame(final, columns=['x','y','color'])

def clusters_advanced(samples):
  x1 = np.random.uniform(-1, 1, samples)
  y1 = np.random.uniform(0,2, samples)
  z1 = np.zeros(y1.shape)
  m1 = np.zeros(y1.shape)

  red = np.column_stack((x1,y1,z1,m1))

  x2 = np.random.uniform(-1, 1, samples)
  y2 = 4 + np.random.uniform(0,2, samples)
  z2 = np.ones(y2.shape)
  m2 = np.zeros(y2.shape)

  blue = np.column_stack((x2,y2,z2,m2))

  x3 = np.random.uniform(-3, -1, samples)
  y3 = np.random.uniform(0,2, samples)
  z3 = np.ones(y3.shape)*2
  m3 = np.ones(y3.shape)

  green = np.column_stack((x3,y3,z3,m3))

  x4 = np.random.uniform(-3, -1, samples)
  y4 = 4 + np.random.uniform(0,2, samples)
  z4 = np.ones(y4.shape)*3
  m4 = np.ones(y4.shape)

  teal = np.column_stack((x4,y4,z4,m4))

  x5 = np.random.uniform(-3, -1, samples)
  y5 = 2 + np.random.uniform(0,2, samples)
  z5 = np.ones(y5.shape)*4
  m5 = np.ones(y5.shape)

  orange = np.column_stack((x5,y5,z5,m5))

  x6 = np.random.uniform(-1, 1, samples)
  y6 = 2 + np.random.uniform(0,2, samples)
  z6 = np.ones(y6.shape)*5
  m6 = np.ones(y6.shape)*2

  purple = np.column_stack((x6,y6,z6,m6))

  final = np.row_stack((red,blue,green,teal,orange,purple))

  return pd.DataFrame(final, columns=['x','y','color','marker'])

def clusters(samples):
  x1 = np.random.uniform(-1, 1, samples)
  y1 = np.random.uniform(0,2, samples)
  z1 = np.zeros(y1.shape)

  red = np.column_stack((x1,y1,z1))

  x2 = np.random.uniform(-1, 1, samples)
  y2 = 4 + np.random.uniform(0,2, samples)
  z2 = np.ones(y2.shape)

  blue = np.column_stack((x2,y2,z2))

  x3 = np.random.uniform(-3, -1, samples)
  y3 = np.random.uniform(0,2, samples)
  z3 = np.ones(y3.shape)*2

  green = np.column_stack((x3,y3,z3))

  x4 = np.random.uniform(-3, -1, samples)
  y4 = 4 + np.random.uniform(0,2, samples)
  z4 = np.ones(y4.shape)*3

  teal = np.column_stack((x4,y4,z4))

  x5 = np.random.uniform(-3, -1, samples)
  y5 = 2 + np.random.uniform(0,2, samples)
  z5 = np.ones(y5.shape)*4

  orange = np.column_stack((x5,y5,z5))

  x6 = np.random.uniform(-1, 1, samples)
  y6 = 2 + np.random.uniform(0,2, samples)
  z6 = np.ones(y6.shape)*5

  purple = np.column_stack((x6,y6,z6))

  final = np.row_stack((red,blue,green,teal,orange,purple))

  return pd.DataFrame(final, columns=['x','y','color'])

color_dict = {0: 'red', 1: 'blue', 2: 'green', 3:'teal', 4:'orange', 5:'purple'}
marker_dict = {0: '^', 1: '+', 2:'*'}

train_df = clusters_advanced(1000)
train_df['color'] = train_df.color.apply(lambda x: color_dict[int(x)])
train_df['marker'] = train_df.marker.apply(lambda x: marker_dict[int(x)])

test_df = clusters_advanced(200)
test_df['color'] = test_df.color.apply(lambda x: color_dict[int(x)])
test_df['marker'] = test_df.marker.apply(lambda x: marker_dict[int(x)])

graph = 'clusters_two_categories'
results_dir = f'./examples/{graph}'
data_dir = f'./examples/{graph}/data'

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

train_df.to_csv(f'{data_dir}/train.csv', index=False)
test_df.to_csv(f'{data_dir}/test.csv', index=False)

#plt.scatter(train_df.x, train_df.y, color=train_df.color, s=2)

for index, row in train_df.iterrows():
	if index%250 == 0:
		print(index)
	plt.scatter(row.x, row.y, color=row.color, marker=row.marker, s=20)



#plt.savefig(f'{results_dir}/figure.png')