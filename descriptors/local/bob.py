import numpy as np
from ML_SK.plotting.plots import plot_matrix

from .smearing import smr1

from collections import OrderedDict
from itertools   import combinations








def _cmatrix_molecular(umol):
    """
    Returns the Molecular BOB vector.
    """
    z_list = np.array(list(OrderedDict.fromkeys(umol.z)))

    zii=np.array([ np.sort(np.array([z1*z1/umol.distance(iz1,iz2)
                                     for k,iz1 in enumerate(np.where(umol.z==z1)[0])
                                     for   iz2 in           np.where(umol.z==z1)[0][:k]]))
                                     for z1 in z_list])

    zij=np.array([ np.sort(np.array([z1*z2/umol.distance(iz1,iz2)
                                     for iz1 in np.where(umol.z==z1)[0]
                                     for iz2 in np.where(umol.z==z2)[0]]) )
                                     for z1,z2 in list(combinations(z_list,2))])

    return np.concatenate((np.hstack(zii),np.hstack(zij)))






def _cmatrix_atomic(umol,ei):
  """
  Returns the BOB vector for each of the
  atoms specified by 'ei'.
  """
  cm=np.zeros([len(ei),umol.natm,umol.natm])

  for cc,i in enumerate(ei):
      dl=np.linalg.norm(umol.R-umol.R[i],axis=1)
      #dl[np.where(dl>3)]=1e5  #Hard cutoff
      dl=dl*smr1(dl,2.,6.,2.)
      sorted_order=np.lexsort((dl,umol.z))

      zz=umol.z[sorted_order]

      dl[i]=1.
      invdl=umol.z[i]/np.power(dl,1.)[sorted_order]
      cm[cc][0]=invdl*zz

  #plot_matrix(cm[0][:1,:19]);  #plot_matrix(cm[1])
  return cm






def _cmatrix_nabla_molecular(umol,direction=0,i=0):
  """
  xxx.
  """
  cm=np.zeros([umol.natm,umol.natm])

  dlv=umol.R-umol.R[i]
  dl =np.linalg.norm(dlv,axis=1)
  dl[i]=1.
  order=range(len(dl))
  zz=umol.z[order]

  for ii in order:
    cm[ii][i]=zz[ii]*zz[i]*dlv[:,direction][ii]/np.power(dl[ii],3.)
    cm[i][ii]=zz[ii]*zz[i]*dlv[:,direction][ii]/np.power(dl[ii],3.)
    cm[i][i]=0.0

  #plot_matrix(cm)
  return -cm[np.triu_indices(cm.shape[0])]






def _cmatrix_nabla_atomic(umol,direction=0,i=0):
  """
  In some point one can choose if it is better the derivative or
  the the unitary directions only.
  """
  cm=np.zeros([umol.natm,umol.natm])

  dlv=umol.R-umol.R[i]
  dl =np.linalg.norm(dlv,axis=1)

  sorted_order=np.lexsort((dl,umol.z))
  dl[i]=1.
  zz=umol.z[sorted_order]#[range(umol.N)]

  sorted_order_ang=np.lexsort((dlv[:,direction]/np.power(dl,1.),umol.z))
  #print '- ',dlv[:,direction]/np.power(dl,1.),' ',
  #print '+ ',dlv[:,direction][sorted_order_ang]/np.power(dl,1.)[sorted_order_ang],; raw_input()

  cm[0]=umol.z[i]*zz[i]*dlv[:,direction][sorted_order_ang]/np.power(dl,1.)[sorted_order_ang]#/smr1(dl,2.,6.,2.)

  #cm[0][0]=0.0

  return -cm[np.triu_indices(cm.shape[0])]


def _BAK_cmatrix_nabla_atomic(umol,direction=0,i=0):
  """
  In some point one can choose if it is better the derivative or
  the the unitary directions only.
  """
  cm=np.zeros([umol.natm,umol.natm])

  dlv=umol.R-umol.R[i]
  dl =np.linalg.norm(dlv,axis=1)

  sorted_order=np.lexsort((dl,umol.z))
  dl[i]=1.
  zz=umol.z[sorted_order]#[range(umol.N)]

  cm[0]=umol.z[i]*zz[i]*dlv[:,direction]/np.power(dl,3.)

  #cm[0][0]=0.0

  return -cm[np.triu_indices(cm.shape[0])]




def _cmatrix_force_atomic(umol,ei,direction=0):
  """
  Returns the vectorized Coulomb Matrix for each of the
  atoms of elemet specified by 'ei'.
  """
  cm=np.zeros([len(ei),umol.natm,umol.natm])

  for cc,i in enumerate(ei):
      #sorting atoms with respect to the head atom
      dl=np.linalg.norm(umol.R-umol.R[i],axis=1)
      sorted_order=np.lexsort((dl,umol.z))

      zz=umol.z[sorted_order]

      for cci,ii in enumerate(sorted_order):
          dlv=umol.R[sorted_order]-umol.R[ii]
          dl=np.linalg.norm(dlv,axis=1)
          dl[cci]=1.
          invdl=umol.z[ii]/np.power(dl,1.)
          invdl=invdl*dlv[:,direction]
          cm[cc][cci]=invdl*zz
          cm[cc][cci][cci]=np.array([0.5*np.power(umol.z[ii],1.4)])

  #plot_matrix(cm[0][:10,:10])
  #plot_matrix(cm[1])
  return cm
  #return np.array([cm[i][np.triu_indices(umol.N)] for i in range(cm.shape[0])])






# Classes ##########################################

class M_Molecular(object):
  def f(self,umol):
    return _cmatrix_molecular(umol)


class M_Atomic(object):
  def f(self,umol,ei, aco='DUMMY_COMPATIBILITY'):
    cm=_cmatrix_atomic(umol,ei)
    return np.array([cm[i][np.triu_indices(umol.N)] for i in range(cm.shape[0])])


class M_Nabla_Molecular(object):
  def f(self,umol,direction=0,i=0):
    return _cmatrix_nabla_molecular(umol,direction,i)


class M_Nabla_Atomic(object):
  def f(self,umol,direction=0,i=0):
    return _cmatrix_nabla_atomic(umol,direction,i)


class M_Force_Atomic(object):
  def f(self,umol,ei, aco='DUMMY_COMPATIBILITY',direction=0):
    cm=_cmatrix_force_atomic(umol,ei,direction)
    return np.array([cm[i][np.triu_indices(umol.N)] for i in range(cm.shape[0])])
