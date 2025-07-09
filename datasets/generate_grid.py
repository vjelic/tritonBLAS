#!/usr/bin/env python3
import yaml
import argparse


def generate_entries(transA, transB, in_dtype, out_dtype):
    """
    Generate all benchmark entries for each combination of dimensions:
      - Each dimension (m, n, k) runs from 128 to 8192 in steps of 128.
      - in_dtype and out_dtype are passed as arguments.
      - transA and transB are passed as arguments and are fixed for all entries.
    """
    entries = []

    for m in range(128, 8192 + 1, 128):
        for n in range(128, 8192 + 1, 128):
            for k in range(128, 8192 + 1, 128):
                entries.append(
                    {
                        "in_dtype": in_dtype,
                        "out_dtype": out_dtype,
                        "transA": transA,
                        "transB": transB,
                        "m": m,
                        "n": n,
                        "k": k,
                        "transA": str(transA),
                        "transB": str(transB),
                    }
                )
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate a YAML file with benchmark sizes for matmul using dimensions from 128 to 8192 in steps of 128, including transpose options and data types."
    )
    parser.add_argument(
        "--transA",
        type=str,
        choices=["N", "T"],
        default="T",
        help="Transpose type for A matrix ('N' for no transpose, 'T' for transpose, default: 'N').",
    )
    parser.add_argument(
        "--transB",
        type=str,
        choices=["N", "T"],
        default="N",
        help="Transpose type for B matrix ('N' for no transpose, 'T' for transpose, default: 'N').",
    )
    parser.add_argument(
        "--in-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16", "float8_e5m2", "float8_e4m3fnuz"],
        default="float16",
        help="Input data type ('float16', 'bfloat16', 'float8_e5m2', 'float8_e4m3fnuz' default: 'float16').",
    )
    parser.add_argument(
        "--out-dtype",
        type=str,
        choices=["float32", "float16", "bfloat16", "float8_e5m2"],
        default="float16",
        help="Output data type ('float32', 'float16', 'bfloat16', 'float8_e5m2', default: 'float16').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="matmul_grid.yaml",
        help="Output YAML filename (default: matmul_grid.yaml).",
    )
    args = parser.parse_args()

    # Create a customized output filename based on transA, transB, in_dtype, and out_dtype
    output_filename = (
        f"matmul_grid_{args.transA}{args.transB}_{args.in_dtype}_{args.out_dtype}.yaml"
    )

    # Generate dataset based on the transpose and data type options
    dataset = generate_entries(args.transA, args.transB, args.in_dtype, args.out_dtype)

    # Save the dataset to the output YAML file
    with open(output_filename, "w") as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {len(dataset)} entries and saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
