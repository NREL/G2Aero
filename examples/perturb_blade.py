import numpy as np
import os

from g2aero.yaml_info import YamlInfo
from g2aero.Grassmann_interpolation import GrassmannInterpolator
from g2aero.PGA import PGAspace
from g2aero.transform import TransformBlade
from g2aero.Grassmann import landmark_affine_transform

output_folder = os.path.join(os.getcwd(), 'Perturbations_blade', )
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

n_landmarks = 401 # has to be 401 or need to remake the database (currently it has 401)

# baseline airfoils for NREL5MW blade
shapes_filename = os.path.join(os.getcwd(), '../', 'data', 'blades_yamls', 'nrel5mw_ofpolars.yaml')
Blade = YamlInfo(shapes_filename, n_landmarks=n_landmarks)
eta_nominal = Blade.eta_nominal
M_yaml = Blade.M_yaml_interpolator
b_yaml = Blade.b_yaml_interpolator
b_pitch = Blade.pitch_axis

shapes_bs = Blade.xy_landmarks
shapes_gr_bs, M_bs, b_bs = landmark_affine_transform(shapes_bs)

# load shapes from the database
shapes_folder = os.path.join(os.getcwd(), '../', 'data', 'airfoils_database', )
airfoils = ['NACA64_A17', 
            'DU21_A17', 'DU25_A17', 'DU30_A17', 'DU35_A17', 'DU40_A17', 'DU00-W2-350',
            'FFA-W3-211',  'FFA-W3-241', 'FFA-W3-270blend', 'FFA-W3-301', 'FFA-W3-330blend', 'FFA-W3-360', 'SNL-FFA-W3-500', ]
files = [os.path.join(shapes_folder, f'{af}.npz') for af in airfoils]

shapes = np.empty((0, 401, 2))
for i, file in enumerate(files):
    one_file_shapes = np.load(file)['shapes']
    shapes = np.vstack((shapes, one_file_shapes))

#### PGA perturbations
pga, _ = PGAspace.create_from_dataset(shapes, n_modes=4)
pga.radius /= 2
new_shapes, coef = pga.generate_perturbed_blade(shapes_gr_bs[2:], n=10)
shapes_gr_circles = np.repeat(shapes_gr_bs[np.newaxis, :2], 10, axis=0)
new_shapes = np.hstack((shapes_gr_circles, new_shapes))

#### if need physical perturbed nominal airfoils (if want to create yaml files)
# new_phys = np.empty_like(new_shapes)
# for j, sample in enumerate(new_shapes):
#   for i, shape in enumerate(sample):
#       new_phys[j, i] = shape @ M_bs[i].T + b_bs[i]

eta_span = np.linspace(0, 1, 200)
for j, sample in enumerate(new_shapes):
    print(f'Sample {j}')
    Grassmann = GrassmannInterpolator(eta_nominal, sample)
    M = Grassmann.interpolator_M
    b = Grassmann.interpolator_b
    _, gr_crosssections = Grassmann(eta_span, grassmann=True)
    #print('\t-Inverse Transform')
    Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
    xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)
    np.savez(os.path.join(output_folder, f'xyz_blade{j}.npz'), xyz=xyz_local, coef=coef[j])

