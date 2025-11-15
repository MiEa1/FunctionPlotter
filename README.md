# Function Plotter

A small, dependency-light command-line function plotter (no GUI).
Generates publication-quality PNG/JPEG plots from mathematical expressions using NumPy, SymPy, Matplotlib, and Pillow.

## Features

* Parse and safely evaluate mathematical expressions using SymPy.
* High-quality plotting with Matplotlib and configurable output resolution.
* Support for multiple functions in a single plot, automatic or manual y-axis scaling.
* Customizable appearance: image size, DPI, background color, line width, colors, grid, axes and ticks.
* CLI-first design suitable for scripting, automation, and headless servers.

## Install

Install the minimal Python dependencies:

```bash
pip install numpy matplotlib sympy pillow
```

This script targets Python 3.10+ and has been tested with Python 3.12.

## Usage

Basic usage:

```bash
python plot_functions.py "sin(x)" "x**2"
```

Common options:

```text
-o, --output      Output image path (default: result.jpg)
-x, --xrange      X-axis range: xmin xmax (default: -10 10)
-y, --yrange      Y-axis range: ymin ymax (default: auto)
-s, --size        Image pixel size: width height (default: 1200 800)
-d, --dpi         Image resolution (default: 100)
--no-axes         Hide axes
--no-ticks        Hide ticks/labels
--tick-step       Tick spacing (default: 1.0)
--colors          List of line colors (e.g. red #00aaff)
--no-grid         Hide grid
--bg-color        Background color (default: white)
--line-width      Line thickness (default: 2.0)
--samples         Samples per x-unit (controls smoothness)
```

Advanced examples:

```bash
# Save a high-resolution plot of multiple functions
python plot_functions.py "sin(x)" "cos(2*x)" -o trig_plot.png -x -6.28 6.28 -s 2400 1600 -d 200 --tick-step 0.5

# Plot a rational function with custom y-range
python plot_functions.py "1/(x-2)" -o pole.png -x -10 10 -y -10 10
```

## Safety and limitations

* Expression parsing is restricted to a whitelist of SymPy math functions to mitigate arbitrary code execution.
* Some pathological expressions or extremely high sampling rates may be slow or memory intensive.
* Complex-valued results are rendered as gaps in the plot to avoid misleading output.

## Development and Contribution

Contributions, bug reports, and feature suggestions are welcome. Please open issues or pull requests with reproducible examples. When adding new functionality, include tests and update the README with new CLI examples.

## License

MIT License — feel free to reuse and adapt for personal or commercial projects.

<!-- WORD_COUNT: 352 -->
