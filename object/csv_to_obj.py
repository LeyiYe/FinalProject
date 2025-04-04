import numpy as np
from scipy.spatial import ConvexHull
import argparse
import os

def convert_csv_to_obj(input_csv, output_obj):
    """Convert a particle CSV file to OBJ format using convex hull"""
    try:
        # Load particle data (skip header if present)
        data = np.loadtxt(input_csv, delimiter=',', skiprows=1 if has_header(input_csv) else 0)
        
        # Generate convex hull
        hull = ConvexHull(data)
        
        # Write OBJ file
        with open(output_obj, 'w') as f:
            # Write vertices
            f.write("# OBJ file generated from particle CSV\n")
            np.savetxt(f, data, fmt='v %.6f %.6f %.6f')
            
            # Write faces (1-based indexing)
            f.write("\n# Faces\n")
            for simplex in hull.simplices:
                f.write(f"f {simplex[0]+1} {simplex[1]+1} {simplex[2]+1}\n")
        
        print(f"Success: {input_csv} → {output_obj} ({len(data)} particles, {len(hull.simplices)} faces)")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

def has_header(csv_path):
    """Check if CSV has a header row"""
    with open(csv_path, 'r') as f:
        first_line = f.readline()
    return not any(char.isdigit() for char in first_line.split(',')[0].strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert particle CSV to OBJ mesh')
    parser.add_argument('input', help='Input CSV filename (without .csv extension)')
    parser.add_argument('output', help='Output OBJ filename (without .obj extension)')
    args = parser.parse_args()

    # Add extensions if not present
    input_csv = args.input if args.input.endswith('.csv') else f"{args.input}.csv"
    output_obj = args.output if args.output.endswith('.obj') else f"{args.output}.obj"

    # Verify input exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found")
        exit(1)

    convert_csv_to_obj(input_csv, output_obj)