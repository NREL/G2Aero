import gmsh
import os
import numpy as np


def blade_CAD_geometry(shapes, outfilename, msh=False):

    gmsh.initialize()
    gmsh.model.add("blade")

    n_shapes, n_landmarks, _ = shapes.shape
    n_half = int(n_landmarks / 2)
    points = np.empty((n_shapes, n_landmarks), dtype=np.int32)
    curve_ind = []
    print('\t-creating cross sections Bsplines')
    for i, shape in enumerate(shapes):
        for j, xyz in enumerate(shape):
            gmsh.model.occ.addPoint(xyz[0], xyz[1], xyz[2], 1, i*n_landmarks+j)
            points[i, j] = (i*n_landmarks) + j
        gmsh.model.occ.addBSpline(points[i, :n_half + 1], i*3)      # lower part
        gmsh.model.occ.addBSpline(points[i, n_half:], i*3+1)        # upper part
        gmsh.model.occ.addLine(points[i, -1], points[i, 0], i*3+2)  # trail edge
        gmsh.model.occ.addCurveLoop([i*3, i*3+1, i*3+2], i)         # connect cross section
        curve_ind.append(i)
        gmsh.model.occ.synchronize()
    # Spanwise Bsplines
    # for j in range(n_landmarks):
    #     gmsh.model.occ.addBSpline(points[:, j], 40001 + j)
    #     gmsh.model.occ.addWire([40001 + j], 50001 + j)
    # gmsh.model.occ.synchronize()
    print('\t-creating ThruSections')
    gmsh.model.occ.addThruSections(curve_ind, makeSolid=True, makeRuled=False)
    gmsh.option.setNumber("Geometry.NumSubEdges", 100)
    gmsh.model.occ.synchronize()
    gmsh.write(outfilename + '.stp')
    # generate mesh
    if msh:
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        # We can constraint the min and max element sizes to stay within reasonable values
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
        gmsh.model.mesh.generate(2)
        gmsh.write(outfilename + '.msh')
    gmsh.finalize()


def write_geofile(shapes, outfilename, n_spanwise=300, n_te=3, n_cross_half=100):
    n_shapes, n_landmarks, _ = shapes.shape
    n_half = int(n_landmarks / 2)

    print('\t-writing .geo file')
    with open(outfilename + '.geo', 'w') as geo:
        geo.write('SetFactory("OpenCASCADE");\n')
        for i, shape in enumerate(shapes):
            for j, xyz in enumerate(shape):
                geo.write(f'Point({i * n_landmarks + j})={{ {xyz[0]}, {xyz[1]}, {xyz[2]}, 1}};\n')
            geo.write(f'Spline({i * 2})={{{i * n_landmarks}:{i * n_landmarks + n_half}}};\n')
            geo.write(f'Spline({i * 2 + 1})={{{i * n_landmarks + n_half}:{(i+1) * n_landmarks - 1}}};\n')
            geo.write(f'Line({10000 + i})={{{(i+1) * n_landmarks - 1}, {i * n_landmarks}}};\n')
            geo.write(f'Curve Loop({i})={{{i * 2}, {i * 2 + 1}, {10000 + i}}};\n\n')
        geo.write(f'\nThruSections(1)={{0:{n_shapes-1}}};\n')

        ind_start = 10000 + n_shapes
        # number of upper and lower cross section points
        geo.write(f'Transfinite Curve{{{ind_start}, {ind_start + 2}, {ind_start + 4}, {ind_start + 6}}} = {n_cross_half} Using Bump -0.01;\n')
        # number of points spanwise
        geo.write(f'Transfinite Curve{{{ind_start + 1}, {ind_start + 3}, {ind_start + 5}}} = {n_spanwise};\n')
        # number of cross points in trail edge
        geo.write(f'Transfinite Curve{{{ind_start + 7}:{ind_start + 8}}} = {n_te};\n')

        geo.write(f'Transfinite Surface{{1}};\n')
        geo.write(f'Transfinite Surface{{2}};\n')
        geo.write(f'Transfinite Surface{{3}};\n')
        geo.write(f'Recombine Surface {{1, 2, 3}};\n')

    os.system(f'gmsh -2 {outfilename}.geo -o {outfilename}_struct.msh')
    print('ind_start', ind_start)




