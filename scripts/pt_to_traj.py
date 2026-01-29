import argparse
import torch
import numpy as np
import random
from ase import Atoms
from ase.io import Trajectory
from ase.geometry import cell_from_parameters

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(args.seed)
    
    print(f"Loading data from {args.input_path}...")
    # Load data to CPU to avoid CUDA OOM during conversion
    data = torch.load(args.input_path, map_location='cpu')
    
    if 'frac_coords' not in data:
        raise ValueError("Input file does not contain 'frac_coords'. Ensure it is a valid CDVAE output file.")

    # Data shapes are typically (num_evals, total_atoms/total_crystals, ...)
    # For optimization output, num_evals is usually 1.
    num_evals = data['num_atoms'].shape[0]
    print(f"Found {num_evals} evaluation set(s). Converting to ASE Atoms objects...")

    all_atoms_list = []

    for eval_idx in range(num_evals):
        frac_coords = data['frac_coords'][eval_idx]
        atom_types = data['atom_types'][eval_idx]
        num_atoms = data['num_atoms'][eval_idx]
        lengths = data['lengths'][eval_idx]
        angles = data['angles'][eval_idx]

        start_idx = 0
        for i in range(len(num_atoms)):
            n_atoms = int(num_atoms[i])
            if n_atoms == 0:
                continue
            
            # Extract properties for the current crystal
            current_frac_coords = frac_coords[start_idx:start_idx + n_atoms]
            current_atom_types = atom_types[start_idx:start_idx + n_atoms]
            
            current_lengths = lengths[i]
            current_angles = angles[i]

            # Create Unit Cell
            cell = cell_from_parameters(
                current_lengths[0].item(),
                current_lengths[1].item(),
                current_lengths[2].item(),
                current_angles[0].item(),
                current_angles[1].item(),
                current_angles[2].item()
            )

            # Create ASE Atoms object
            atoms = Atoms(
                numbers=current_atom_types.numpy(),
                scaled_positions=current_frac_coords.numpy(),
                cell=cell,
                pbc=True
            )
            all_atoms_list.append(atoms)
            
            start_idx += n_atoms

    print(f"Converted {len(all_atoms_list)} structures.")
    print(f"Saving to {args.output_path}...")
    
    traj = Trajectory(args.output_path, 'w')
    for atoms in all_atoms_list:
        traj.write(atoms)
    traj.close()
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .pt file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output .traj file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args)
