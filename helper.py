import numpy as np
import sys
import os
z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])

def indices_periodic_BC(i,j,k,u,d1, d2, d3):
    if u == 0:
        return np.array([[i, j, k, 1], [i, j, k, 2], [i, j, k, 3], [np.mod(i-1, d1), j, k, 1], [i, np.mod(j-1, d2), k, 2], [i, j, np.mod(k-1, d3), 3]])
    elif u == 1:
        return np.array([[i, j, k, 0], [i, j, k, 2], [i, j, k, 3], [np.mod(i+1, d1), j, k, 0],[np.mod(i+1, d1), np.mod(j-1, d2), k, 2], [np.mod(i+1, d1), j, np.mod(k-1, d3), 3]])
    elif u == 2:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 3], [i, np.mod(j+1, d2), k, 0], [np.mod(i-1, d1), np.mod(j+1, d2), k, 1], [i, np.mod(j+1, d2), np.mod(k-1, d3), 3]])
    elif u == 3:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 2], [i, j, np.mod(k+1, d3), 0], [np.mod(i-1, d1), j, np.mod(k+1, d3), 1], [i, np.mod(j-1, d2), np.mod(k+1, d3), 2]])

def indices_open_BC(i,j,k,u,d1, d2, d3):
    if u == 0:
        return np.array([[i, j, k, 1], [i, j, k, 2], [i, j, k, 3], [i-1, j, k, 1], [i, j-1, k, 2], [i, j, k-1, 3]])
    elif u == 1:
        return np.array([[i, j, k, 0], [i, j, k, 2], [i, j, k, 3], [i+1, j, k, 0],[i+1, j-1, k, 2], [i+1, j, k-1, 3]])
    elif u == 2:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 3], [i, j+1, k, 0], [i-1, j+1, k, 1], [i, j+1, k-1, 3]])
    elif u == 3:
        return np.array([[i, j, k, 0], [i, j, k, 1], [i, j, k, 2], [i, j, k+1, 0], [i-1, j, k+1, 1], [i, j-1, k+1, 2]])



dim1 = int(sys.argv[9])
dim2 = int(sys.argv[10])
dim3 = int(sys.argv[11])


con = np.zeros((dim1, dim2, dim3, 4))

Jxx, Jyy, Jzz = float(sys.argv[1]), float(sys.argv[2]),float(sys.argv[3])

Jpm = -(Jxx+Jyy)/4
Jpmpm = (Jxx-Jyy)/4
h = float(sys.argv[4])
fielddir = np.array([float(sys.argv[5]),float(sys.argv[6]), float(sys.argv[7])])
fielddir = fielddir/np.linalg.norm(fielddir)
B = np.einsum('r, ir->i', h*fielddir,z)


def flattenIndex(Indx):
    temp = np.zeros(len(Indx))
    for i in range(6):
        temp[i] = Indx[i][0]*dim2*dim3*4 + Indx[i][1]*dim3*4 + Indx[i][2]*4 + Indx[i][3]
    return temp

def genNN_list(d1,d2,d3, PBC = True):
    NN_list = np.zeros((d1*d2*d3*4, 6))
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                for u in range(4):
                    if PBC:
                        NN_list[i*d2*d3*4+j*d3*4+k*4+u] = flattenIndex(indices_periodic_BC(i,j,k,u,d1,d2,d3))
                    else:
                        NN_list[i*d2*d3*4+j*d3*4+k*4+u] = flattenIndex(indices_open_BC(i,j,k,u,d1,d2,d3))
    return NN_list

look_up_table = genNN_list(dim1, dim2, dim3)

#Sz = 2, Sp = 0, Sm = 1
def HeisenbergNN(Jzz, Jpm, Jpmpm, indx1, indx2):
    if indx1 <= dim1*dim2*dim3*4 and indx2 <= dim1*dim2*dim3*4 and indx1 >= 0 and indx2 >= 0:
        Jzz = Jzz/2 
        Jpm = Jpm/2
        Jpmpm = Jpmpm/2
        return np.array([[2, indx1, 2, indx2, Jzz, 0],
                        
                        [0, indx1, 1, indx2, -Jpm, 0],
                        [1, indx1, 0, indx2, -Jpm, 0],

                        [1, indx1, 1, indx2, Jpmpm, 0],
                        [0, indx1, 0, indx2, Jpmpm, 0]])

def Zeeman(h, indx):
    here = h[indx % 4]
    return np.array([[2, indx, -here, 0]])   


interALL = []
transfer = []


for i in range(len(look_up_table)):
    transfer.append(Zeeman(B, i))
    for j in range(len(look_up_table[i])):
        interALL.append(HeisenbergNN(Jzz, Jpm, Jpmpm, i, look_up_table[i][j]))

interALL = np.array(interALL).reshape(-1, 6)
transfer = np.array(transfer).reshape(-1, 4)




output_dir = "./" + sys.argv[8] + "/"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

max_site = dim1*dim2*dim3*4
All_N = max_site
exct = 1

fstrength = np.zeros((1,1))
fstrength[0,0] = h
np.savetxt(output_dir+"field_strength.dat", fstrength)


def write_interALL(interALL, file_name):
    num_param = len(interALL)
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("num "+"{0:8d}".format(num_param)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for i in range(num_param):
        f.write(" {0:8d} ".format(int(interALL[i,0])) \
        +" {0:8d}   ".format(int(interALL[i,1]))     \
        +" {0:8d}   ".format(int(interALL[i,2]))     \
        +" {0:8d}   ".format(int(interALL[i,3]))     \
        +" {0:8f}   ".format(interALL[i,4]) \
        +" {0:8f}   ".format(interALL[i,5]) \
        +"\n")
    f.close()

def write_transfer(interALL, file_name):
    num_param = len(interALL)
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("num "+"{0:8d}".format(num_param)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for i in range(num_param):
        f.write(" {0:8d} ".format(int(interALL[i,0])) \
        +" {0:8d}   ".format(int(interALL[i,1]))     \
        +" {0:8f}   ".format(interALL[i,2])     \
        +" {0:8f}   ".format(interALL[i,3])
        +"\n")
    f.close()

write_interALL(interALL, 'InterAll.def')
write_transfer(transfer, 'Trans.def')
