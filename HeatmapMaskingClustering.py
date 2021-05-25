import numpy as np
def a2l(arr,ele_dim):
    arr = np.array(arr)
    s = arr.shape
    if s[len(s)-1] != ele_dim:
        raise Exception()
    return np.reshape(arr,[-1,ele_dim])

def l2a(list,sample_arr):
    s1 = sample_arr.shape
    s2 = list.shape
    newshape = np.concatenate([s1[:len(s1)-1],s2[len(s2)-1:]])
    return np.reshape(list,newshape)

class ndMatrix:
    def __init__(self,A):
        self.data = np.reshape(A,[-1])
        self.shape = np.array(A.shape)
        self.multiplier = [np.prod(np.concatenate([self.shape[i+1:],np.array([1])])) for i in range(len(self.shape))]
        self.multiplier = np.array(self.multiplier,dtype = np.int64)

        self.vecM = lambda A: np.reshape(A,[np.prod(self.SHAPE['input shape']),1])
        self.vecN = lambda B: np.reshape(B,[np.prod(self.SHAPE['output shape']),1])

        self.devecM = lambda a: np.reshape(a,self.SHAPE['input shape'])
        self.devecN = lambda b: np.reshape(b,self.SHAPE['output shape'])

    def len(self):
        return len(self.data)

    def ij2k(self,ij_arr):
        ij_arr = np.array(ij_arr)
        ij_list = a2l(ij_arr,len(self.shape))
        mul = np.broadcast_to(self.multiplier,[ij_list.shape[0],self.multiplier.shape[0]])
        k_list = np.sum(ij_list * mul, axis = 1, dtype = np.int64, keepdims = True)
        k_arr = l2a(k_list,ij_arr)
        return k_arr

    def k2ij(self,k_arr):
        k_arr = np.array(k_arr)
        k_list = a2l(k_arr,1)
        ij_list = np.zeros([len(k_list),self.multiplier.shape[0]])
        for i,m in enumerate(self.multiplier):
            ij_list[:,i:i+1] = np.array(k_list / m, dtype = np.int64)
            k_list = k_list % m
        ij_arr = l2a(ij_list,k_arr)
        return ij_arr

    def print(self):
        return np.reshape(self.data,self.shape)
    
    def copy(self):
        return ndMatrix(self.print())

    def is_valid(self,ij_arr):
        return np.logical_and(np.array(ij_arr) < self.shape,0 <= np.array(ij_arr))

    def get_by_k_arr(self,k_arr):
        k_arr = np.array(k_arr)
        k_list = a2l(k_arr,1)
        Ak_list = self.data[k_list]
        Ak_arr = l2a(Ak_list,k_arr)
        return Ak_arr

    def set_by_k_arr(self,k_arr,x_arr):
        k_arr,x_arr = np.array(k_arr),np.array(x_arr)
        s = x_arr.shape
        k_list,x_list = a2l(k_arr,1),a2l(x_arr,s[len(s)-1])
        self.data[k_list] = x_list

    def get(self,ij_arr):
        ij_arr = np.array(ij_arr)
        valid_ij = self.is_valid(ij_arr)
        ij_arr = ij_arr * valid_ij
        k_arr = self.ij2k(ij_arr)
        return self.get_by_k_arr(k_arr) * np.prod(valid_ij, axis = len(valid_ij.shape)-1, keepdims=True)
        
    def set(self,ij_arr,x_arr):
        ij_arr,x_arr = np.array(ij_arr),np.array(x_arr)
        k_arr = self.ij2k(ij_arr)
        self.set_by_k_arr(k_arr,x_arr)
        
        
    from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from scipy.signal import convolve2d as conv
import scipy.stats as st
import numpy as np

def gkern(kernlen, nsig):
        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()


def QiSeg(B,iterparam,epoch):
        C = np.array(B)
        for i in range(epoch):
            b1,b2,sig,kernlen = iterparam(i)
            filter = gkern(kernlen,sig)
            C = conv(C,filter,'same','fill')
            C = C*(C>b1)

            L,n = ndimage.label(C)
            D = np.zeros(C.shape)
            for i in range(1,n+1):
                E = C*(L==i)
                D = D + E/np.max(E)
            C = D

            C = C*(C>b2)

        L,n = ndimage.label(C)
        return L

class HMC:
    def __init__(self, left_bondaries, right_bondaries, heatmap_shape,
                 bias1 = 0.1, bias2 = 0.1, sigma = 2, kernlen = 5, epoch = 10):
        self.left_bondaries = np.array(left_bondaries)
        self.right_bondaries = np.array(right_bondaries)
        self.heatmap_shape = np.array(heatmap_shape)
        self.heatmap = np.zeros(heatmap_shape)

        self.bias1 = bias1
        self.bias1 = bias2
        self.sigma = sigma
        self.kernlen = kernlen
        self.epoch = epoch

    def to_indices(self,x_batch):
        n = len(x_batch)
        d = len(self.left_bondaries)

        x = np.array(x_batch)
        l = np.broadcast_to(self.left_bondaries,(n,d))
        r = np.broadcast_to(self.right_bondaries,(n,d))
        s = np.broadcast_to(self.heatmap_shape,(n,d))

        idx = np.array((x-l)/(r-l)*(s-2)+1,dtype = np.int32)
        idx = idx * (idx>=0)
        idx = idx * (idx<(s-1)) + (s-1)*(idx>=(s-1))

        return idx
    
    def feed(self,data):
        IDX = self.to_indices(data)
        ndH = ndMatrix(self.heatmap)
        for idx in IDX:
            ndH.set([idx],ndH.get([idx])+1)
            self.heatmap = ndH.print()

    def segment(self):
        iterparam = lambda i: (self.bias1**(i+1),self.bias1**(i+1),self.sigma**i,self.kernlen)
        self.mask = QiSeg(1-1/np.exp2(self.heatmap),iterparam,self.epoch)
        self.partition = np.array(self.mask)
        A = np.array(self.mask)
        def nm(x,y):
            return np.sqrt(x*x+y*y)
        X,Y = np.where(A==0)
        for i in range(len(X)):
            x,y,D = X[i],Y[i],[]
            for l in range(1,int(np.max(A))+1):
                XX,YY = np.where(A==l)
                D.append(np.min(nm(XX-x,YY-y)))
            self.partition[x,y] = np.argmin(np.array(D))+1

    def predict(self,data,datatype = 'global'):
        IDX = self.to_indices(data)
        x,y = IDX.transpose()
        if datatype == 'global':
            return self.partition[x,y]
        return self.mask[x,y]
