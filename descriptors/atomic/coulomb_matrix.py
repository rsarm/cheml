import numpy as np
from ML_SK.plotting.plots import plot_matrix




def _cmatrix_molecular(umol):
  """
  Returns the vectorized Coulomb Matrix.
  """
  cm=np.zeros([umol.natm,umol.natm])
  order=np.array(range(umol.natm))

  zz=umol.z[order]

  for i in range(umol.natm):
    dl=np.linalg.norm(umol.R[order]-umol.R[i],axis=1)
    dl[i]=1.
    invdl=umol.z[i]/dl
    cm[i]=invdl*zz
    cm[i][i]=np.array([0.5*np.power(umol.z[i],2.4)])

  return cm[np.triu_indices(cm.shape[0])]




def _cmatrix_atomic(umol,ei,cut_off=80.):
  """
  Returns the Coulomb Matrix for each of the
  atoms specified by 'ei'.
  """
  cm=np.zeros([len(ei),umol.natm,umol.natm])

  for cc,i in enumerate(ei):
      #sorting atoms with respect to the head atom
      dl=np.linalg.norm(umol.R-umol.R[i],axis=1)

      cmlen=len(dl[np.where(dl<cut_off)])
      sorted_order=dl.argsort()[:cmlen]

      if cc==0: sorted_order=[ 0,  1,  9,  2, 10,  5,  3, 14, 12,  6, 13,  7, 11,  4, 16, 15,  8, 17, 18]
      if cc==1: sorted_order=[ 4,  3,  5, 13, 12, 11,  2,  6, 14, 10,  9,  1,  8, 17,  7, 18, 16,  0, 15]

      zz=umol.z[sorted_order]

      for cci,ii in enumerate(sorted_order):
          dl=np.linalg.norm(umol.R[sorted_order]-umol.R[ii],axis=1)
          dl[cci]=1.
          invdl=umol.z[ii]/dl
          cm[cc][cci][:cmlen]=invdl*zz
          cm[cc][cci][cci]=np.array([0.5*np.power(umol.z[ii],2.4)])

      #plot_matrix(cm[cc])
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
  sorted_order=dl.argsort()
  dl[i]=1.
  zz=umol.z[range(umol.N)]

  if i==0: sorted_order=[ 0,  1,  9,  2, 10,  5,  3, 14, 12,  6, 13,  7, 11,  4, 16, 15,  8, 17, 18]
  if i==4: sorted_order=[ 4,  3,  5, 13, 12, 11,  2,  6, 14, 10,  9,  1,  8, 17,  7, 18, 16,  0, 15]


  for ii in sorted_order:
    cm[ii][i]=zz[ii]*zz[i]*dlv[:,direction][ii]/np.power(dl[ii],3.)
    cm[i][ii]=zz[ii]*zz[i]*dlv[:,direction][ii]/np.power(dl[ii],3.)
    #cm[ii][i]=dlv[:,direction][ii]/dl[ii]
    #cm[i][ii]=dlv[:,direction][ii]/dl[ii]
    cm[i][i]=0.0

  #plot_matrix(cm[sorted_order].T[sorted_order])
  return -cm[sorted_order].T[sorted_order][np.triu_indices(cm.shape[0])]



def _cmatrix_force_atomic(umol,ei,direction=0):
  """
  Returns the vectorized Coulomb Matrix for each of the
  atoms of elemet specified by 'ei'.
  """
  cm=np.zeros([len(ei),umol.natm,umol.natm])

  for cc,i in enumerate(ei):
      #sorting atoms with respect to the head atom
      dl=np.linalg.norm(umol.R-umol.R[i],axis=1)

      sorted_order=dl.argsort()

      #if i==0: sorted_order=[ 0,  1,  9,  2, 10,  5,  3, 14, 12,  6, 13,  7, 11,  4, 16, 15,  8, 17, 18]
      #if i==4: sorted_order=[ 4,  3,  5, 13, 12, 11,  2,  6, 14, 10,  9,  1,  8, 17,  7, 18, 16,  0, 15]

      zz=umol.z[sorted_order]

      for cci,ii in enumerate(sorted_order):
          dlv=umol.R[sorted_order]-umol.R[ii]
          dl=np.linalg.norm(dlv,axis=1)

          dl[cci]=1.
          invdl=umol.z[ii]/np.power(dl,1.)
          invdl=invdl*dlv[:,direction]
          cm[cc][cci]=invdl*zz
          cm[cc][cci][cci]=np.array([0.5*np.power(umol.z[ii],2.4)])

  #plot_matrix(cm[0][:10,:10]); plot_matrix(cm[1])
  return cm







# Classes ##########################################

class M_Molecular(object):
  def f(self,umol):
    return _cmatrix_molecular(umol)


class M_Atomic(object):
  def f(self,umol,ei):
    cm=_cmatrix_atomic(umol,ei)
    return np.array([cm[i][np.triu_indices(umol.N)] for i in range(cm.shape[0])])


class M_Nabla_Molecular(object):
  def f(self,umol,direction=0,i=0):
    return _cmatrix_nabla_molecular(umol,direction,i)


class M_Nabla_Atomic(object):
  def f(self,umol,direction=0,i=0):
    return _cmatrix_nabla_atomic(umol,direction,i)


class M_Force_Atomic(object):
  def f(self,umol,ei,direction=0):
    cm=_cmatrix_force_atomic(umol,ei,direction)
    return np.array([cm[i][np.triu_indices(umol.N)] for i in range(cm.shape[0])])
