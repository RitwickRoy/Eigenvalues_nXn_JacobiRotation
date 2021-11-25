# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 22:57:06 2021

@author: Ritwick
"""

#
#  Compute Eigenvalues of a n X n  symmetric matrix using Jacobi Rotation
#  Jacobi Rotation: Applies a series of transformations to diagonalize the 
#                   given matrix. Each transformation is a rotation that
#                   operates on a 2X2 sub-matrix. Only the upper-diagonal
#                   part of the matrix is treated due to the symmetry.
#  ref: Numerical Linear Algebra, L.N. Trefenthen and D. Bau. (Lecture 30)

import numpy as np


class JacobiRotation:
    def __init__(self,S):
        self.S = S
        self.nrows = self.S.shape[0]
        self.ncols = self.S.shape[1]
        self.theta = 0.0
        self.i = 0
        self.j = 0
        self.tol = 0.000001
        self.tol_iter = 0.001
        self.niter = 3
        self.maxdiag = -1.7976931348623157e+308
        self.maxoffdiag = -1.7976931348623157e+308

    #
    #  Compute the angle for Givens Rotation that 
    #  zeros the off-diagonal term of 2X2 sub-matrix after rotation
    #
    def ComputeTheta(self):
        anum = 2.0*self.S[self.i][self.j]
        aden = self.S[self.j][self.j] - self.S[self.i][self.i]
        # Avoid division by zero
        if abs(aden) > self.tol:
            self.theta = 0.5*np.arctan(anum/aden)
        else:
            self.theta = np.pi/4.0

    #
    #  Update the terms of the 2X2 sub-matrix after applying the rotation. 
    #  Update the max diagonal and off-diagonal term from this sub-matrix
    #
    def DiagUpdate(self):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        c2 = c*c
        s2 = s*s
        sc = c*s
        self.S[self.i][self.i] = c2*self.S[self.i][self.i] - 2.0*sc*self.S[self.i][self.j] + s2*self.S[self.j][self.j]
        self.S[self.j][self.j] = s2*self.S[self.i][self.i] + 2*sc*self.S[self.i][self.j] + c2*self.S[self.j][self.j]
        self.S[self.i][self.j] = (c2-s2)*self.S[self.i][self.j] + sc*(self.S[self.i][self.i]-self.S[self.j][self.j])
        self.maxdiag = max(self.maxdiag,abs(self.S[self.i][self.i]),abs(self.S[self.j][self.j]))
        self.maxoffdiag = max(self.maxoffdiag, abs(self.S[self.i][self.j]))
    
    #
    #  To avoid searchin for the largest off-diagonal term, rotations are applied to
    #  all  2X2 sub-matrices sequentially (1,2),(1,3)... 
    #  treating only the upper diagonal terms.
    #
    def CyclicSweep(self):
        for l in range(self.nrows-1):
            for m in range(l+1, self.ncols):
                # select the 2X2 submatrix for Givens rotation
                self.i = l
                self.j = m
                self.ComputeTheta()
                self.DiagUpdate()
    #
    #  Iterate through rotation sweeps until the convergence.
    #  The ratio of the largest off-diagonal to the largest diagonal
    #  term for each sweep is checked against tolerance level.
    #
    def Diagonalize(self):
        ratio_offdiag_diag = self.maxoffdiag/self.maxdiag
        while ratio_offdiag_diag > self.tol_iter:
            # reinitialize max diagonal and off-diagonal for this sweep
            self.maxdiag = -1.7976931348623157e+308
            self.maxoffdiag = -1.7976931348623157e+308
            self.niter += 1
            self.CyclicSweep()
            ratio_offdiag_diag = self.maxoffdiag/self.maxdiag

    #
    #  The diagonal terms after convergence are the eigenvalues.
    #
    def OutputEigenValues(self):
        print('number of iterations for convergence:',self.niter)
        print()
        for i in range(self.nrows):
            print('Eig',i,':',self.S[i][i])

######################################################
#
#  Main driver for testing:
#
#  1. generate a  random n X n matrix
#  2. symmetrize the random matrix
#  3. Find eigenvalues of the matrix using JacobiRotation
#  4. Check the eigenvalues numpy computed eigenvalues
#
######################################################

u_str = input('Input the size of the matrix: ')
n = int(u_str)
b = np.random.random((n,n))
#
#  symmetrize the matrix 
#  a = 1/2 *(b + b_transpose)
#
a = 0.5*(b + np.transpose(b))
#
print()
print('Random', n,'X',n,' symmetric matrix')
print()
print(a)
print()
jr = JacobiRotation(a)
jr.Diagonalize()
jr.OutputEigenValues()
print()
print('Check eigenvalues against numpy:')
print('Note: eigenvalues are in different order')
print()
e =np.linalg.eigvals(a)
print(e)