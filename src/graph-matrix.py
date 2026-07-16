from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

import scienceplots


plt.style.use(['science','grid' ]) 


matrix10 = [[0.0916, 0.    , 0.    , 0.   ],
             [0.    , 0.4755, 0.3257, 0.   ],
             [0.    , 0.3257, 0.4329, 0.   ],
             [0.    , 0.    , 0.    , 0.   ]]
matrix20 = [[0.0509, 0.    , 0.    , 0.   ],
             [0.    , 0.5852, 0.3508, 0.   ],
             [0.    , 0.3508, 0.3638, 0.   ],
             [0.    , 0.    , 0.    , 0.   ]]

matrix50= [[0.0651, 0.,     0.,     0.    ],
             [0.,     0.4327, 0.3885, 0.    ],
             [0.,     0.3885, 0.4987, 0.    ],
             [0.,     0.,     0.,     0.0035]]
matrix1000= [[0.004,  0.,     0.,     0.    ],
             [0.,     0.493,  0.4735, 0.    ],
             [0.,     0.4735, 0.496,  0.    ],
             [0.,     0.,     0.,     0.007 ]]

for bits in [10, 20, 50, 1000]:
    for flip in [True, False]:
        matrix = matrix50 if bits == 50 else matrix20 if bits == 20 else matrix10
        z_data = np.array(matrix)
        x_size, y_size = z_data.shape

        import numpy as np

        if flip == True:
            # Swap rows using an index array
            z_data[[0, 1, 2, 3]] = z_data[[1, 0, 3, 2]]

            # Swap columns using an index array on the second axis
            z_data[:, [0, 1, 2, 3]] = z_data[:, [1, 0, 3, 2]]



        # Initialize a perfectly square canvas
        fig = plt.figure(figsize=(5, 5))

        # Add 3D axes directly to the figure (no subplot/gridspec grids)
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7], projection='3d', proj_type='ortho')

        _x = np.arange(x_size)
        _y = np.arange(y_size)
        _xx, _yy = np.meshgrid(_x, _y)

        x_flat = (_xx.ravel()).astype(float)
        y_flat = (_yy.ravel()).astype(float)
        z_flat = np.zeros_like(x_flat)

        dx = 0.8
        dy = 0.8
        x_flat -= dx/2
        y_flat -= dy/2
        dz = z_data.ravel()

        # Create floor tile scaling
        dz_plot = np.where(dz > 0, dz, 0.0001)

        colormap = plt.cm.YlGnBu 
        norm = colors.Normalize(vmin=0, vmax=0.6)
        rgba_colors = np.zeros((len(dz), 4))

        for i, val in enumerate(dz):
            if val > 0:
                rgba_colors[i] = colormap(norm(val))
                rgba_colors[i, 3] = 0.9 
            else:
                rgba_colors[i] = [0.88, 0.88, 0.88, 0.25] 

        # Render everything in a single call
        ax.bar3d(x_flat, y_flat, z_flat, dx, dy, dz_plot, 
                 color=rgba_colors,
                 edgecolor='k', 
                 linewidth=0.3,
                 shade=True)

        # Clean up axes, remove background panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 0.4)
        ax.yaxis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 0.4)
        ax.zaxis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 0.4)


        labels = ['HH', 'HV', 'VH', 'VV'] # if flip == False else ['HH', 'HV', 'VH', 'HH']
        ax.set_xticks(_x, labels)
        ax.set_yticks(_y, labels)
        ax.set_zlim(0, 0.6)
        ax.set_zticks([0, 0.2, 0.4, 0.6])

        ax.set_box_aspect((1, 1, 1), zoom=0.85)

        ax.view_init(elev=30, azim=45)
        if flip == True:
            plt.title(f"Density Matrix for {bits} bits, H/V Basis")

        else:
            plt.title(f"Density Matrix for {bits} bits, D/A Basis")

        # Save without tight crop engine to prevent clipping

        if flip == True:
            plt.savefig(f"connection_matrix_{bits}bits_HHVV.png", dpi=300, bbox_inches=None)
        else:
            plt.savefig(f"connection_matrix_{bits}bits.png", dpi=300, bbox_inches=None)