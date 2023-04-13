import numpy as np
import os
from g2aero.SPD import polar_decomposition

from g2aero.yaml_info import YamlInfo
from g2aero.Grassmann_interpolation import GrassmannInterpolator
from g2aero.PGA import Grassmann_PGAspace
from g2aero.transform import TransformBlade, global_blade_coordinates
from g2aero.SPD import polar_decomposition

examples_path = os.path.dirname(__file__)
root_path = os.path.abspath(os.path.join(examples_path, os.pardir))


output_folder = os.path.join(examples_path, 'Perturbations_blade', )
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

N = 10 # Number of perturbation to generate

# load shapes from the database
shapes_file = os.path.join(root_path, 'data', 'airfoils', 'CST_shapes_TE_gap.npz')
shapes = np.load(shapes_file)['shapes']
# define n_landmarks based on dataset data
n_landmarks = shapes.shape[1]

#make PGA space
pga, _ = Grassmann_PGAspace.create_from_dataset(shapes, n_modes=4)
pga.radius /= 2

# load baseline airfoils for a blade from .yaml file
shapes_filename = os.path.join(root_path, 'data', 'blades_yamls', 'IEA-15-240-RWT.yaml')
Blade =  YamlInfo.init_from_yaml(shapes_filename, n_landmarks=n_landmarks)
eta_nominal = Blade.eta_nominal
M_yaml = Blade.M_yaml_interpolator
b_yaml = Blade.b_yaml_interpolator
b_pitch = Blade.pitch_axis
# given (baseline) blade airfoils
shapes_bs = Blade.xy_landmarks[2:]  # skip hub circles
shapes_gr_bs, M_bs, b_bs = polar_decomposition(shapes_bs)

# perturb the blade
new_blades, coef = pga.generate_perturbed_blade(shapes_gr_bs, n=N)

# Inverse affine transform to get physical perturbed nominal airfoils
new_phys_blades = np.empty_like(new_blades)
for i, new_blade in enumerate(new_blades):
    for j, sh in enumerate(new_blade):
        new_phys_blades[i, j] = sh @ M_bs[j] + b_bs[j]

# Add hub circles back
shapes_circles = np.repeat(Blade.xy_landmarks[np.newaxis, :2], N, axis=0)
new_phys_blades = np.hstack((shapes_circles, new_phys_blades))

# Interpolate perturbed blades
for j, new_phys_blade in enumerate(new_phys_blades):
    GrInterp = GrassmannInterpolator(eta_nominal, new_phys_blade)
    M = GrInterp.interpolator_M
    b = GrInterp.interpolator_b
    eta_span = GrInterp.sample_eta(100, n_hub=10, n_tip=10, n_end=25)
    _, gr_crosssections = GrInterp(eta_span, grassmann=True)
    Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
    xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)
    xyz_global = global_blade_coordinates(xyz_local)
    np.savez(os.path.join(output_folder, f'xyz_blade{j}.npz'), xyz=xyz_local, coef=coef[j])

