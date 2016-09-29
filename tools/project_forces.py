
# Script to calculate the projections of the force or vectorial 3d properties in general
# on the interatomic distances weighted by different functions.

import sys
import numpy as np

import read_xyz as rxyz  #only needed for write_formated.

from cheml.io                   import xyz
from cheml                      import dataset, _DATA_FOLDER




def decay(r,function=None):
    if function== None: return 1.
    if function=='cos':
        d=r.dot(r)
        #if d<36.: return 0.5*(1+np.cos(np.pi*np.sqrt(d)/6.))
        if d<25.: return (1-np.sin(np.pi*np.sqrt(d)/10.))
        else: return 0.

def Rij(i,j):
    """Vector Z_i*Z_j/R_ij^2 from atom j to atom i"""
    if i==j: return np.array([0.,0.,0.])
    r=(mol.R[i]-mol.R[j])
    return r*np.power(r.dot(r),-2.)#*mol.z[i]*mol.z[j]


def Rmatrix(t):
    """Matrix of the unitary distance vectors
    from all the molecules to a target atom
    """
    coordlist=np.zeros(mol.natm)
    c=0
    for ia in range(mol.natm):
        #if ia!=t:
        coordlist[c]=ia
        c+=1
    return  np.array([Rij(t,ia)*decay(Rij(t,ia),function=None)
                      for ia in coordlist.astype(int) ])


def prj_print(mol,component):
    print mol.N,'\n','Mol', mol.energy, '0.0'
    for i in range(mol.natm):
        print mol.symb[i],
        print rxyz.write_formated( mol.R[i] ),
        print rxyz.write_formated( component[:,i] )

ds=dataset.dataset()
ds.read_xyz(sys.argv[1])


for mol in ds.list_of_mol:
  Ft=np.zeros([mol.N,3])
  for i in range(mol.N):
      Ft[i]=mol.data[i]

  R =np.array([Rmatrix(i) for i in range(mol.N)])
  fc=np.empty((mol.natm,mol.natm))

  for i in range(mol.N):
      fc[i,:]=np.einsum('ik,k->i',R[i],Ft[i,:])

  prj_print(mol,fc)




exit()
#Reconstruction of the force
print ''
mol=ds.list_of_mol[0]
for i in range(mol.N):
    RTdotR =np.einsum('ki,kj->ij',R[i],R[i])
    RTdotRinv =np.linalg.inv(RTdotR)
    RTdotfc=np.einsum('ki,k->i'  ,R[i],fc[i,:])
    print int(mol.z[i]),
    print rxyz.write_formated( mol.R[i] ),'    ',
    #print rxyz.write_formated(np.einsum('ik,k->i',RTdotRinv,RTdotfc))
    #check that the cartesian forces are correct:
    print rxyz.write_formated(np.einsum('ik,k->i',RTdotRinv,RTdotfc)-mol.data[i])










exit()
def _Rmatrix(t):
    """Matrix of the unitary distance vectors
    from all the molecules to a target atom
    """
    coordlist=np.zeros(mol.natm-1)
    c=0
    for ia in range(mol.natm):
        if ia!=t:
            coordlist[c]=ia
            c+=1
    return  np.array([Rij(t,ia)*decay(Rij(t,ia),function=None) 
                      for ia in coordlist.astype(int) ])


for mol in ds.list_of_mol:
  Ft=np.zeros([mol.N,3])
  for i in range(mol.N):
      Ft[i]=mol.data[i]

  R =np.array([_Rmatrix(i) for i in range(mol.N)])
  fc=np.empty((mol.natm,mol.natm-1))

  for i in range(mol.N):
      fc[i,:]=np.einsum('ik,k->i',R[i],Ft[i,:])

  print mol.N,'\n','Mol 0.0 0.0'
  print mol.symb[0],rxyz.write_formated(mol.R[0]),' 0.0000000'
  for i in range(mol.natm-1):
      print mol.symb[i+1],
      print rxyz.write_formated( mol.R[i+1] ),'  ',
      #print rxyz.write_formated( fc[:,i] )
      print "%15.10f" % (fc[0,i])
