import numpy as np
import pandas as pd
import tensorflow as tf
# df = pd.DataFrame([[1,18,3],[4,5,6],[7,8,9]],columns=['a','b','c'])
# df1 = pd.DataFrame([[11,21,31],[41,51,61],[71,81,91]],columns=['a1','b1','c1'])
# df2 = pd.concat([df,df1],axis=1)
# df['d'] = [7,8,9]
# a =df.sort_index(axis=0,ascending=False)
# print()
# print(df['a'].loc[1:2])
# ser = df.iloc[0]-df.iloc[1]
# print(type(df.iloc[0]-df.iloc[1]))
# ser['date'] = "2016-09-08"
# print(ser)
# a = [np.array([[1],
#        [1],
#        [1],
#        [1],
#        [1],
#        [1],
#        [1]])]
#
# print(a[0].reshape(len(a[0]),))

#
# print(int(5/2))

# a = [1,2,3,4,5]
# a.pop()
# print(a)



a = tf.zeros([3, 5, 7, 9])

with tf.compat.v1.Session() as sess:
    _a = sess.run(a)
    print(_a)



