import sys

def compare(file_a, file_b):

    files = ["B_10.vtk", "E_10.vtk", "rho_net_10.vtk", "rhoe_10.vtk", "rhoi_10.vtk"]

    for f in files:
        with open(f"{file_a}/data/{f}") as f_a, open(f"{file_b}/data/{f}") as f_b:
                for x,y in zip(f_a,f_b):
                    try:
                        fx = float(x)
                        fy = float(y)
                        if(abs(fx-fy) > 1e-6):
                            print(f"{fx} and {fy} differ by {abs(fx-fy)}")
                    except:
                        if(x != y):
                            print(x, y)

compare(sys.argv[1], sys.argv[2])
