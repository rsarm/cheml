import numpy as np

from scipy.spatial import distance





Z={'X'  :0,  'Ar' :18.,
   'H'  :1., 'C'  : 6., 'N'  :7., 'O'  :8. , 'F'  :9., 'Cu'  :29.,
   '1.0':1., '6.0': 6., '7.0':7., '8.0':8. , '9.0':9., '29.0':29.,
   '18.0':18., 'Cl': 17., '17.0':17.,'S':16.,'16.0':16.,
   'P':15, '15.0':15.
  }


str_z2s={'1.0':'H', '6.0':'C', '7.0':'N', '8.0':'O', '9.0':'F', '18.0':'Ar','29.0':'Cu','17.0':'Cl','16.0':'S','15.0':'P'}

