import pyvista as pv

# Read the parallel VTI file

# Read the parallel VTI file
mesh = pv.read("Florian1d1q3_sod_Rho,U,Theta,Entropy_P00_00000358.vti")

# Display basic information or plot the mesh
print(mesh)
mesh.plot()

rho = mesh["Rho"]
u = mesh["U"]
theta = mesh["Theta"]
entropy = mesh["Entropy"]
pressure = mesh["Pressure"]

print(rho.shape)   # check shape — scalar field vs vector field (U might be (N,3))

points = mesh.points          # (N, 3) array of x, y, z coordinates
x = points[:, 0]              # 1D case: just take the x-coordinate
u_x = u[:, 0]
centers = mesh.cell_centers().points
x = centers[:, 0]
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0,0].plot(x, rho);    axs[0,0].set_title("Rho")
axs[0,1].plot(x, u_x);    axs[0,1].set_title("U")
axs[1,0].plot(x, theta);  axs[1,0].set_title("Theta")
axs[1,1].plot(x, entropy);axs[1,1].set_title("Entropy")
plt.tight_layout()
plt.show()
