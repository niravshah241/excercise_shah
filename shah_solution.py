import numpy as np
from scipy.integrate import dblquad
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import sympy

def eq_solve(mat_A, vec_b, solver=None):
	'''Solves linear system of equation using specified solver
	Input:
	mat_A (numpy array) : Matrix, with applied boundary conditions, corresponding to bilinear form
	vec_b (numpy array) : vector, with applied boundary conditions, corresponding to linear form
	Output:
	sol_u (numpy array) : returns vector containing degrees of freedom'''
	
	if solver==None:
		sol_u = np.linalg.solve(mat_A, vec_b)
	elif solver=="Cholesky":
		sol_u = cho_solve(cho_factor(mat_A),vec_b)
	elif solver=="LU":
		sol_u = lu_solve(lu_factor(mat_A),vec_b)
	else:
		raise Error("Solver not available")
	return sol_u

def eval_elemental_matrix(x_step, y_step, pol_degree=1):
	'''Computes the matrix entries of bilinear form at a given elemental_matrix
	Input:
	x_step (float) : step size in x directions
	y_step (float) : step size in y directions
	pol_degree (int) : Polynomial degree of interpolation (Currently only degree 1 implemented)
	Output:
	elemental_matrix (numpy array) : Matrix corresponding to discretised bilinear form'''
	
	elemental_matrix = np.zeros([4,4]) # currently only polynomial degree 1 implemented
	x = sympy.Symbol("x")
	y = sympy.Symbol("y")
	phi_a = (1-x)*(1-y)
	phi_b = x * (1-y)
	phi_d = (1-x) * y
	phi_c = x * y
	elemental_matrix[0][0] = x_step * y_step * sympy.integrate(sympy.diff(phi_a,x)*sympy.diff(phi_a,x)/(x_step**2)+sympy.diff(phi_a,y)*sympy.diff(phi_a,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix[0][1] = x_step * y_step * sympy.integrate(sympy.diff(phi_a,x)*sympy.diff(phi_b,x)/(x_step**2)+sympy.diff(phi_a,y)*sympy.diff(phi_b,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix[0][2] = x_step * y_step * sympy.integrate(sympy.diff(phi_a,x)*sympy.diff(phi_d,x)/(x_step**2)+sympy.diff(phi_a,y)*sympy.diff(phi_d,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix[0][3] = x_step * y_step * sympy.integrate(sympy.diff(phi_a,x)*sympy.diff(phi_c,x)/(x_step**2)+sympy.diff(phi_a,y)*sympy.diff(phi_c,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix[1][1] = x_step * y_step * sympy.integrate(sympy.diff(phi_b,x)*sympy.diff(phi_b,x)/(x_step**2)+sympy.diff(phi_b,y)*sympy.diff(phi_b,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix[1][2] = x_step * y_step * sympy.integrate(sympy.diff(phi_b,x)*sympy.diff(phi_d,x)/(x_step**2)+sympy.diff(phi_b,y)*sympy.diff(phi_d,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix[1][3] = x_step * y_step * sympy.integrate(sympy.diff(phi_b,x)*sympy.diff(phi_c,x)/(x_step**2)+sympy.diff(phi_b,y)*sympy.diff(phi_c,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix[2][2] = x_step * y_step * sympy.integrate(sympy.diff(phi_d,x)*sympy.diff(phi_d,x)/(x_step**2)+sympy.diff(phi_d,y)*sympy.diff(phi_d,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix[2][3] = x_step * y_step * sympy.integrate(sympy.diff(phi_d,x)*sympy.diff(phi_c,x)/(x_step**2)+sympy.diff(phi_d,y)*sympy.diff(phi_c,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix[3][3] = x_step * y_step * sympy.integrate(sympy.diff(phi_c,x)*sympy.diff(phi_c,x)/(x_step**2)+sympy.diff(phi_c,y)*sympy.diff(phi_c,y)/(y_step**2),(x,0,1),(y,0,1))
	elemental_matrix = elemental_matrix + elemental_matrix.T - np.diag(np.diag(elemental_matrix))
	return elemental_matrix

# Discretsation of domain: Domain is discretised with nx equidistant nodes in x-direction and ny equidistant nodes in y-direction. The corresponding step size in x-direction is hx=1/(nx-1) and step size in y-direction is hy=1/(ny-1). The cooordinates (x,y) of node in row i and column j is given by x = i*hx and y = i*hy.
nx, ny = 7,5#100, 90 #number of steps in x and y directions
hx, hy = 1./(nx-1), 1./(ny-1) #step sizes in x and y directions
x, y = np.linspace(0,1,nx), np.linspace(0,1,ny) # Mesh points in x and y directions
xx, yy = np.meshgrid(x, y, sparse=True)

# Extract boundary nodes
boundary_nodes_indices = list()
for i in range(nx):
	boundary_nodes_indices.append(i)
for j in range(1,ny-1):
	boundary_nodes_indices.append(nx*j)
	boundary_nodes_indices.append(nx*j+nx-1)
for i in range(nx):
	boundary_nodes_indices.append(nx*(ny-1)+i)

# Create vector of boundry values. The nodes not on boundary are set as zero.
boundary_values = np.zeros(nx*ny)
boundary_values[boundary_nodes_indices] = 0.

# Manually coded elemental matrix for simplicity, eval_elemental_matrix (see above) computes same matrix with symbolic computations
local_mat = \
	np.array([[(1./3.)*(hy/hx+hx/hy), (1./3.)*(hx/(2*hy)-hy/hx), (1./3.)*(hy/(2*hx)-hx/hy), (-1./6.)*(hy/hx+hx/hy)],\
	[(1./3.)*(hx/(2*hy)-hy/hx), (1./3.)*(hy/hx+hx/hy), (-1./6.)*(hy/hx+hx/hy), (1./3.)*(hy/(2*hx)-hx/hy)],\
	[(1./3.)*(hy/(2*hx)-hx/hy), (-1./6.)*(hy/hx+hx/hy), (1./3.)*(hy/hx+hx/hy), (1./3.)*(hx/(2*hy)-hy/hx)],\
	[(-1./6.)*(hy/hx+hx/hy), (1./3.)*(hy/(2*hx)-hx/hy), (1./3.)*(hx/(2*hy)-hy/hx), (1./3.)*(hy/hx+hx/hy)]]) #for pol_degree=1

# Alternate to manual code elemental matrix using smbolic computations (see function eval_elemental_matrix)
local_mat_sympy = eval_elemental_matrix(hx,hy,pol_degree=1)

print(local_mat)
print(local_mat_sympy)

# Assembly of discretised bilinear form. Precisely below block places the elemental_matrix into global matrix based on global index of the node. Notice the advantage of symmetry during assembly.
a_matrix2 = np.zeros([nx*ny,nx*ny]) # a_matrix is global bilinear matrix
for j in range(ny-1):
	for i in range(nx-1):
		a_matrix2[i+nx*j,i+nx*j] += local_mat[0,0]
		a_matrix2[i+nx*j,i+nx*j+1] += local_mat[0,1]
		a_matrix2[i+nx*j,i+nx*(j+1)] += local_mat[0,2]
		a_matrix2[i+nx*j,i+nx*(j+1)+1] += local_mat[0,3]
		
		a_matrix2[i+nx*j+1,i+nx*j+1] += local_mat[1,1]
		a_matrix2[i+nx*j+1,i+nx*(j+1)] += local_mat[1,2]
		a_matrix2[i+nx*j+1,i+nx*(j+1)+1] += local_mat[1,3]
		
		a_matrix2[i+nx*(j+1),i+nx*(j+1)] += local_mat[2,2]
		a_matrix2[i+nx*(j+1),i+nx*(j+1)+1] += local_mat[2,3]
		
		a_matrix2[i+nx*(j+1)+1,i+nx*(j+1)+1] += local_mat[3,3]

a_matrix2 = a_matrix2 + a_matrix2.T - np.diag(np.diag(a_matrix2))

# Assembly of discretised linear form. source_func1, source_func2, source_func3 and source_func4 are linear form expressed in local coordinate system
l_vector = np.zeros(nx*ny)
for j in range(ny-1):
	for i in range(nx-1):
		source_func1 = lambda x,y: 4. * (-(hy*y+j*hy)**2+(hy*y+j*hy)) * np.sin(np.pi*(hx*x+i*hx)) * (1-x) * (1-y) * hx * hy 
		l_vector[i+nx*j] += dblquad(source_func1,0,1,lambda x: 0, lambda x: 1)[0]
		source_func2 = lambda x,y: 4. * (-(hy*y+j*hy)**2+(hy*y+j*hy)) * np.sin(np.pi*(hx*x+i*hx)) * x * (1-y) * hx * hy
		l_vector[i+nx*j+1] += dblquad(source_func2,0,1,lambda x: 0, lambda x: 1)[0]
		source_func3 = lambda x,y: 4. * (-(hy*y+j*hy)**2+(hy*y+j*hy)) * np.sin(np.pi*(hx*x+i*hx)) * (1-x) * y * hx * hy
		l_vector[i+nx*(j+1)] += dblquad(source_func3,0,1,lambda x: 0, lambda x: 1)[0]
		source_func4 = lambda x,y: 4. * (-(hy*y+j*hy)**2+(hy*y+j*hy)) * np.sin(np.pi*(hx*x+i*hx)) * x * y * hx * hy
		l_vector[i+nx*(j+1)+1] += dblquad(source_func4,0,1,lambda x: 0, lambda x: 1)[0]

# Application of boundary conditions to discretised linear form. The discretised linear form is modified to take into account modifications in discretised bilinear form to take into account modifications on the discretised bilinear form.
for j in boundary_nodes_indices:
	l_vector -= a_matrix2[:,j] * boundary_values[j]
l_vector[boundary_nodes_indices] = boundary_values[boundary_nodes_indices]

# Application of boundary conditions to discretised bilinear form. The rows and column of a_matrix corresponding to boundary nodes are set to zero with 1 on the diagonal. 
a_matrix2[:,boundary_nodes_indices] = 0
a_matrix2[boundary_nodes_indices,:] = 0
for i in boundary_nodes_indices:
	a_matrix2[i,i] = 1.

#Computation of solution (see function eq_solve)
u = eq_solve(a_matrix2, l_vector)
u_plot = np.reshape(u,(ny,nx))

# Plot solution
plt.figure(figsize=(6,6))
plt.title("Contour plot of solution u")
cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=np.min(u_plot), vmax=np.max(u_plot))
plt.contourf(x,y,u_plot, 100, cmap=cmap)
plt.axis('scaled')
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("solution_u.png")
plt.show()
