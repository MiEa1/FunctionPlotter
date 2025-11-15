#!/usr/bin/env python3
"""
plot_functions.py

GUI-less math plotter.
Supports:
 - Ordinary functions y=f(x)
 - Multiple functions on same image
 - Parametric equations x(t), y(t)
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify
from PIL import Image
import argparse
import os

# ========================
#   Allowed safe functions
# ========================
allowed_names = {
    k: __import__("math").__dict__[k]
    for k in [
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sqrt", "log", "log10", "exp", "pi", "e",
        "fabs"
    ]
}
allowed_names["abs"] = abs

# ============
#   Argument
# ============
parser = argparse.ArgumentParser(description="Function / Parametric plotter")
parser.add_argument("functions", nargs="*", help="Functions y=f(x)")
parser.add_argument("--param", action="store_true", help="Enable parametric mode")
parser.add_argument("--t", nargs=2, type=float, default=[0, 2*np.pi], metavar=('T_MIN', 'T_MAX'))
parser.add_argument("--xrange", nargs=2, type=float, default=[-10, 10])
parser.add_argument("--yrange", nargs=2, type=float, default=[-10, 10])
parser.add_argument("-s", "--size", nargs=2, type=int, default=[800, 600], help="Image size W H")
parser.add_argument("-o", "--output", default="output.jpg", help="Output file name")
args = parser.parse_args()

# ========================
#   Begin plotting
# ========================
plt.figure(figsize=(args.size[0] / 100, args.size[1] / 100))

x = symbols('x')
t = symbols('t')

if args.param:
    # ================
    #   Parametric
    # ================
    if len(args.functions) != 2:
        print("Parametric mode requires exactly two functions: X(t) Y(t)")
        exit(1)

    fx_expr = sympify(args.functions[0], locals=allowed_names)
    fy_expr = sympify(args.functions[1], locals=allowed_names)

    fx = lambdify(t, fx_expr, 'numpy')
    fy = lambdify(t, fy_expr, 'numpy')

    tmin, tmax = args.t
    tp = np.linspace(tmin, tmax, 2000)

    X = fx(tp)
    Y = fy(tp)

    plt.plot(X, Y, linewidth=2)

else:
    # ================
    #   Ordinary y=f(x)
    # ================
    xmin, xmax = args.xrange
    xp = np.linspace(xmin, xmax, 2000)

    for func in args.functions:
        f_expr = sympify(func, locals=allowed_names)
        f = lambdify(x, f_expr, 'numpy')
        try:
            yp = f(xp)
            plt.plot(xp, yp, linewidth=2)
        except Exception as e:
            print(f"Error in function {func}: {e}")

plt.xlim(args.xrange)
plt.ylim(args.yrange)
plt.grid(True)
plt.savefig(args.output, dpi=120)
plt.close()

print(f"Saved to {args.output}")
