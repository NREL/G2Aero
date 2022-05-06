import os
import time
from g2aero.Grassmann import *

def main():
    
    shapes_folder = os.path.join(os.getcwd(), '../', 'data', 'airfoils_database', )
    files = [os.path.join(shapes_folder, file) for file in os.listdir(shapes_folder) if file[-3:]=='npz']

    shapes= np.empty((0, 401, 2))
    for i, file in enumerate(files):
        one_file_shapes = np.load(file)['shapes']
        shapes = np.vstack((shapes, one_file_shapes))
    shapes_gr_all, _, _ = landmark_affine_transform(shapes) 
    
    n = 10000
    t_karcher, t_pga = 0., 0.
    for i in range(10):
        shapes_gr = shapes_gr_all[np.random.choice(shapes_gr_all.shape[0], n, replace=False), :, :]

        t1 = time.perf_counter()
        mu = Karcher(shapes_gr)
        t2 = time.perf_counter()
        t_karcher += t2 - t1
        

        PGA(mu, shapes_gr)
        t3 = time.perf_counter()
        t_pga[i] += t3 - t1
        
    print(f"Karcher mean calculation in {t_karcher/10:0.4f} seconds")
    print(f"Principal Geodesic Analysis in {t_pga/10:0.4f} seconds")

    t_log, t_exp, t_transport = 0., 0., 0.
    for i in range(100):
        X = shapes_gr_all[np.random.randint(len(shapes), size=3)]
        
        t1 = time.perf_counter()
        Delta = log(X[0], X[1])
        t2 = time.perf_counter()
        t_log += t2 - t1

        t1 = time.perf_counter()
        exp(1, X[0], Delta)
        t2 = time.perf_counter()
        t_exp += t2 - t1

        vector = log(X[1], X[2])
        t1 = time.perf_counter()
        parallel_translate(X[0], Delta, vector)
        t2 = time.perf_counter()
        t_transport += t2 - t1
        
    print(f"Log calculation in {t_log/100:0.4f} seconds")
    print(f"Exp calculation in {t_exp/100:0.4f} seconds")
    print(f"Parallel transport calculation in {t_transport/100:0.4f} seconds")


if __name__ == "__main__":
    main()