#!/usr/bin/env python3
"""
plot_functions.py

GUI-less function plotter. Dependencies: numpy, matplotlib, sympy, pillow
Install: pip install numpy matplotlib sympy pillow
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify
from PIL import Image
import math
import os
import sys
from itertools import cycle
import argparse

# Import allowed math functions (security constrained)
from sympy import sin, cos, tan, asin, acos, atan
from sympy import sinh, cosh, tanh, exp, log, sqrt, pi

# Allowed math functions mapping
ALLOWED_FUNCS = {
    'sin': sin, 'cos': cos, 'tan': tan,
    'asin': asin, 'acos': acos, 'atan': atan,
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
    'exp': exp, 'log': log, 'sqrt': sqrt, 'pi': pi
}

# Symbolic variable x
x_symbol = symbols('x')


def plot_functions(
    func_expressions,
    output_path="result.jpg",
    x_range=(-10, 10),
    y_range=None,
    image_size=(1200, 800),  # (width_px, height_px)
    dpi=100,
    show_axes=True,
    show_ticks=True,
    tick_step=1.0,
    colors=None,
    grid=True,
    bg_color='white',
    line_width=2.0,
    samples_per_unit=100
):
    """
    Plot functions and save image.
    
    Args:
        func_expressions: List of function strings (e.g., ["sin(x)", "x**2"])
        output_path: Save path for image
        x_range: (xmin, xmax) for x-axis
        y_range: (ymin, ymax) for y-axis (auto if None)
        image_size: Image pixel dimensions
        dpi: Image resolution
        show_axes: Show x=0/y=0 axes
        show_ticks: Show axis ticks and labels
        tick_step: Spacing between ticks
        colors: Line colors (cycles if insufficient)
        grid: Show grid lines
        bg_color: Background color
        line_width: Line thickness
        samples_per_unit: Sampling density per x-unit (smoother lines)
    """
    # Prepare x values
    x_min, x_max = float(x_range[0]), float(x_range[1])
    if x_max <= x_min:
        raise ValueError("x_range must have xmin < xmax")
    
    total_x_units = x_max - x_min
    num_samples = max(200, int(total_x_units * samples_per_unit))
    x_values = np.linspace(x_min, x_max, num_samples)

    # Parse functions (string → computable)
    parsed_functions = []
    for expr in func_expressions:
        try:
            sym_expr = sympify(expr, locals=ALLOWED_FUNCS)
            func = lambdify(x_symbol, sym_expr, 'numpy')
            parsed_functions.append((expr, func))
        except Exception as e:
            raise ValueError(f"Invalid expression '{expr}': {e}")

    # Setup color cycle
    color_cycle = cycle(colors if colors else 
                       ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

    # Create figure
    width_px, height_px = image_size
    fig = plt.figure(figsize=(width_px/dpi, height_px/dpi), dpi=dpi, facecolor=bg_color)
    ax = fig.add_subplot(1, 1, 1)

    # Evaluate and plot each function
    y_min_list, y_max_list = [], []
    for expr, func in parsed_functions:
        try:
            y_values = func(x_values)
        except Exception:
            # Fallback to element-wise computation
            vec_func = np.vectorize(lambda xx: float(func(xx)))
            y_values = vec_func(x_values)
        
        # Handle complex/invalid values
        y_values = np.array(y_values, dtype=np.complex128)
        mask = np.isfinite(y_values.real) & (np.abs(y_values.imag) < 1e-9)
        y_real = np.where(mask, y_values.real, np.nan)

        # Track y-range for auto-scaling
        finite_y = y_real[np.isfinite(y_real)]
        if finite_y.size > 0:
            y_min_list.append(np.nanmin(finite_y))
            y_max_list.append(np.nanmax(finite_y))

        # Plot line
        ax.plot(x_values, y_real, label=expr, linewidth=line_width, color=next(color_cycle))

    # Set y-range (auto if not specified)
    if y_range is None:
        if y_min_list and y_max_list:
            y_min = min(y_min_list)
            y_max = max(y_max_list)
            y_range_len = y_max - y_min if y_max > y_min else 1.0
            y_min -= 0.08 * y_range_len
            y_max += 0.08 * y_range_len
        else:
            y_min, y_max = -1, 1
    else:
        y_min, y_max = float(y_range[0]), float(y_range[1])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Configure axes
    if show_axes:
        # Center axes
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        # Hide top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Bring axes to front
        ax.spines['left'].set_zorder(3)
        ax.spines['bottom'].set_zorder(3)
        # Plot origin marker
        ax.plot(0, 0, marker='o', markersize=3, color='black', zorder=4)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Configure ticks
    if show_ticks:
        # X ticks
        xt_min = math.ceil(x_min / tick_step) * tick_step
        xt_max = math.floor(x_max / tick_step) * tick_step
        if xt_min <= xt_max:
            xticks = np.arange(xt_min, xt_max + 1e-9, tick_step)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{val:.6f}".rstrip('0').rstrip('.') if '.' in f"{val:.6f}" else f"{val:.0f}" for val in xticks])
        # Y ticks
        yt_min = math.ceil(y_min / tick_step) * tick_step
        yt_max = math.floor(y_max / tick_step) * tick_step
        if yt_min <= yt_max:
            yticks = np.arange(yt_min, yt_max + 1e-9, tick_step)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{val:.6f}".rstrip('0').rstrip('.') if '.' in f"{val:.6f}" else f"{val:.0f}" for val in yticks])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Configure grid
    if grid:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)

    # Adjust layout and save
    plt.tight_layout()

    # Create output directory if needed
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save image
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)

    # Try to open image
    try:
        img = Image.open(output_path)
        img.show()
    except Exception as e:
        print(f"Saved to {output_path}, but failed to open: {e}", file=sys.stderr)

    return output_path


if __name__ == "__main__":
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Function plotter - plot custom functions from command line')
    parser.add_argument('functions', nargs='+', help='Function expressions (e.g., "sin(x)" "x**2")')
    parser.add_argument('-o', '--output', default='result.jpg', help='Output image path')
    parser.add_argument('-x', '--xrange', nargs=2, type=float, default=[-10, 10], help='X-axis range (xmin xmax)')
    parser.add_argument('-y', '--yrange', nargs=2, type=float, default=None, help='Y-axis range (ymin ymax)')
    parser.add_argument('-s', '--size', nargs=2, type=int, default=[1200, 800], help='Image size (width height) in pixels')
    parser.add_argument('-d', '--dpi', type=int, default=100, help='Image resolution')
    parser.add_argument('--no-axes', action='store_false', dest='show_axes', help='Hide axes')
    parser.add_argument('--no-ticks', action='store_false', dest='show_ticks', help='Hide ticks/labels')
    parser.add_argument('--tick-step', type=float, default=1.0, help='Tick spacing')
    parser.add_argument('--colors', nargs='+', default=None, help='Line colors (e.g., red #00aaff)')
    parser.add_argument('--no-grid', action='store_false', dest='grid', help='Hide grid')
    parser.add_argument('--bg-color', default='white', help='Background color')
    parser.add_argument('--line-width', type=float, default=2.0, help='Line thickness')
    parser.add_argument('--samples', type=int, default=100, help='Samples per x-unit')

    # Parse arguments
    args = parser.parse_args()

    # Run plotting
    try:
        out_path = plot_functions(
            func_expressions=args.functions,
            output_path=args.output,
            x_range=tuple(args.xrange),
            y_range=tuple(args.yrange) if args.yrange else None,
            image_size=tuple(args.size),
            dpi=args.dpi,
            show_axes=args.show_axes,
            show_ticks=args.show_ticks,
            tick_step=args.tick_step,
            colors=args.colors,
            grid=args.grid,
            bg_color=args.bg_color,
            line_width=args.line_width,
            samples_per_unit=args.samples
        )
        print(f"Plot saved to: {out_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)