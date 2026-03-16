import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as animation
import os
from ngsolve import x, y, IfPos, CoefficientFunction, exp

# ---------------------------------------------------------------------------
# Source term helpers (return NGSolve CoefficientFunctions)
# ---------------------------------------------------------------------------


def sharp_source(center, width, value=1.0, axis='x'):
    """
    Return a sharp-edged bump profile centred at `center` with width `width`
    along `axis` ('x' or 'y').

    The profile is 1 inside [center - width/2, center + width/2] and 0 outside.
    """
    coord = x if axis == 'x' else y
    return value * IfPos(coord - (center - width / 2),
                         IfPos((center + width / 2) - coord, 1.0, 0.0), 0.0)


def tanh_source(center, width, value=1.0, axis='x', interface_length=0.1):
    """
    Return a smooth tanh-based bump profile centred at `center` with
    characteristic width `width` along `axis` ('x' or 'y').

    The profile is:  value * 0.5 * (tanh((coord - (center - width/2)) / w)
                                   - tanh((coord - (center + width/2)) / w))
    where w = interface_length  (controls the edge sharpness; smaller values = sharper).
    """
    coord = x if axis == 'x' else y
    w = interface_length
    tanh_cf = lambda z: (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    left  = (coord - (center - width / 2)) / w
    right = (coord - (center + width / 2)) / w
    return value * 0.5 * (tanh_cf(left) - tanh_cf(right))

# ---------------------------------------------------------------------------
# General purpose functions
# ---------------------------------------------------------------------------

def nematic_to_vector(Q, q):
    theta = np.arctan2(q, Q)/2 
    s = np.sqrt(q**2 + Q**2)
    n = [s*np.cos(theta), s*np.sin(theta)]
    return np.array(n)

# ---------------------------------------------------------------------------
# Matplotlib plot helpers
# ---------------------------------------------------------------------------

def add_cbar(ax, norm, cmap, vmin, vmax, label, fontsize=13):
    """Add a static colorbar that is never auto-updated by im.set_data()."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])          # empty array → colorbar is independent of image data
    cbar = plt.colorbar(sm, ax=ax, location='bottom', pad=0.02,
                        fraction=0.04, aspect=30, shrink=0.9)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f'{vmin:.2g}', f'{vmax:.2g}'])
    cbar.set_label(label, fontsize=fontsize, labelpad=2)
    return cbar


def _auto_fields(data):
    """Return a default list of (key, label, cmap) panel specs for *data*."""
    fields = [('rho', r'$\rho$', 'inferno')]
    if 'm' in data:
        fields.append(('m', r'$m$ (myosin)', 'viridis'))
    fields.append(('v', r'$|v|$', 'Reds'))
    if 'div_v' in data:
        fields.append(('div_v', r'$-\nabla \cdot v$', 'RdBu_r'))
    fields.append(('Q', r'$Q_{yy}$', 'RdBu_r'))
    return fields


def _arrow_idx(n_size, n_arrows):
    """Return ``n_arrows`` evenly-spaced integer indices into [0, n_size)."""
    return np.round(np.linspace(0, n_size - 1, n_arrows)).astype(int)


def _draw_panel(ax, key, label, cmap, data, t, extent, Xc, Yc, n_arrows,
                fontsize=13, vlims=None, ix=None, iy=None):
    """Render a single panel on *ax*.

    Returns (im, qv) — the AxesImage and Quiver artists — so the animation
    helper can update them in-place without redrawing colorbars each frame.

    Parameters
    ----------
    n_arrows : int
        Number of arrow grid points along each axis (gives n_arrows × n_arrows
        evenly-spaced arrows).
    vlims : dict, optional
        Pre-computed ``{key: (vmin, vmax)}`` for consistent colour ranges
        across frames (used by the animation helper).
    ix, iy : array-like of int, optional
        Pre-computed x/y index arrays (avoids recomputing on every animation
        frame).  Derived from ``n_arrows`` when omitted.
    """
    symmetric = 'RdBu' in cmap
    qv = None

    def _get(k):
        arr = data[k]
        return arr[t] if arr.ndim == 3 else arr

    if key == 'v':
        vx, vy = _get('vx'), _get('vy')
        vmag = np.sqrt(vx**2 + vy**2) + 1e-10
        vmin_v, vmax_v = (vlims['v'] if vlims and 'v' in vlims
                          else (0, float(vmag.max())))
        norm = Normalize(vmin=vmin_v, vmax=vmax_v, clip=True)
        im = ax.imshow(vmag.T, origin='lower', cmap=cmap,
                       extent=extent, norm=norm)
        if vmag.max() > 1e-10 and Xc is not None:
            _ix = ix if ix is not None else _arrow_idx(vx.shape[0], n_arrows)
            _iy = iy if iy is not None else _arrow_idx(vx.shape[1], n_arrows)
            qv = ax.quiver(Xc[np.ix_(_iy, _ix)], Yc[np.ix_(_iy, _ix)],
                           (vx / vmag)[np.ix_(_ix, _iy)].T,
                           (vy / vmag)[np.ix_(_ix, _iy)].T,
                           scale=20, width=0.008, pivot='mid',
                           headwidth=3, headlength=3, color='black')
        add_cbar(ax, norm, cmap, round(vmin_v, 3), round(vmax_v, 3),
                 label, fontsize=fontsize)

    elif key == 'Q':
        Q_f = _get('Q')
        if vlims and 'Q' in vlims:
            vmax_q = vlims['Q'][1]
        else:
            vmax_q = max(float(np.abs(Q_f).max()), 1e-6)
        norm = Normalize(vmin=-vmax_q, vmax=vmax_q, clip=True)
        im = ax.imshow((-Q_f).T, origin='lower', cmap=cmap,
                       extent=extent, norm=norm)
        if 'q' in data and Xc is not None:
            q_f = _get('q')
            nx_vec, ny_vec = nematic_to_vector(Q_f, q_f)
            nmag = np.sqrt(nx_vec**2 + ny_vec**2)
            if nmag.max() > 1e-10:
                _ix = ix if ix is not None else _arrow_idx(Q_f.shape[0], n_arrows)
                _iy = iy if iy is not None else _arrow_idx(Q_f.shape[1], n_arrows)
                qv = ax.quiver(
                    Xc[np.ix_(_iy, _ix)], Yc[np.ix_(_iy, _ix)],
                    (nx_vec / nmag)[np.ix_(_ix, _iy)].T,
                    (ny_vec / nmag)[np.ix_(_ix, _iy)].T,
                    scale=20, width=0.01, pivot='mid',
                    headwidth=0, headlength=0, headaxislength=0,
                    color='black')
        add_cbar(ax, norm, cmap, -round(vmax_q, 2), round(vmax_q, 2),
                 label, fontsize=fontsize)

    else:
        field = _get(key)
        if vlims and key in vlims:
            lo, hi = vlims[key]
        elif symmetric:
            vm = max(float(np.abs(field).max()), 1e-6)
            lo, hi = -vm, vm
        else:
            lo, hi = 0.0, max(float(np.max(field)), 1e-6)
        norm = Normalize(vmin=lo, vmax=hi, clip=True)
        im = ax.imshow(field.T, origin='lower', cmap=cmap,
                       extent=extent, norm=norm, interpolation='none')
        add_cbar(ax, norm, cmap, round(lo, 2), round(hi, 2),
                 label, fontsize=fontsize)

    ax.set_xticks([]); ax.set_yticks([])
    return im, qv


def plot_2d_frame(data, t=-1, fields=None, n_arrows=10, title='',
                  filename=None, fontsize=13):
    """Unified 2D frame plot with configurable panels.

    Parameters
    ----------
    data : dict
        Export dict with keys like 'rho', 'vx', 'vy', 'Q', 'q', 'x', 'y', ...
    t : int
        Time-frame index (-1 = last frame).
    fields : list of (key, label, cmap), optional
        Each tuple defines one panel.  Special keys:
          'v'  – velocity magnitude + normalised arrows (uses 'vx', 'vy')
          'Q'  – nematic Q_yy (= −Q_xx) + director arrows (uses 'Q', 'q')
        A cmap containing 'RdBu' is automatically centred at 0.
        If *None*, auto-detected from data keys.
    n_arrows : int
        Number of arrow grid points along each axis (n_arrows × n_arrows
        evenly-spaced arrows).
    title : str
        Figure suptitle.
    filename : str, optional
        Save figure to this path.
    fontsize : int
        Colorbar label font size.
    """
    if fields is None:
        fields = _auto_fields(data)

    n_panels = len(fields)
    extent = (data['x'][0], data['x'][-1],
              data['y'][0], data['y'][-1]) if 'x' in data else (0, 1, 0, 1)
    Xc, Yc = (np.meshgrid(data['x'], data['y'])
              if 'x' in data else (None, None))

    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.2),
                             sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (key, label, cmap) in zip(axes, fields):
        _draw_panel(ax, key, label, cmap, data, t, extent, Xc, Yc,
                    n_arrows, fontsize)

    if title:
        fig.suptitle(title, y=1.02)
    fig.subplots_adjust(wspace=0.02, left=0.02, right=0.98, bottom=0.12, top=0.95)
    if filename is not None:
        plt.savefig(filename, dpi=400)
    plt.show()


def animate_2d(data, filename='animation.mp4', fields=None,
               n_arrows=10, fps=8, fontsize=11, dpi=120,
               title_fmt='t = {t:.1f}', dt=1.0):
    """Create an animated MP4 (or GIF) from a time-series data dict.

    Parameters
    ----------
    data : dict
        Export dict (same format as ``plot_2d_frame``).  Time-dependent
        arrays must have shape ``(N, nx, ny)``.
    filename : str
        Output path.  Supports ``.mp4`` (default, requires ffmpeg) and
        ``.gif`` (uses Pillow).
    fields : list of (key, label, cmap), optional
        Panel specs (see ``plot_2d_frame``).  Auto-detected if *None*.
    n_arrows : int
        Number of arrow grid points along each axis (n_arrows × n_arrows
        evenly-spaced arrows).
    fps : int
        Frames per second.
    fontsize : int
        Colorbar label font size.
    dpi : int
        Resolution of each frame.
    title_fmt : str
        Format string for suptitle; receives ``t`` (time) and ``i`` (index).
        Set to ``''`` to disable.
    dt : float
        Physical time between saved frames (used in *title_fmt*).
    """
    if fields is None:
        fields = _auto_fields(data)

    N = next(data[k].shape[0] for k in data if data[k].ndim == 3)
    n_panels = len(fields)
    extent = (data['x'][0], data['x'][-1],
              data['y'][0], data['y'][-1]) if 'x' in data else (0, 1, 0, 1)
    Xc, Yc = (np.meshgrid(data['x'], data['y'])
              if 'x' in data else (None, None))

    # Pre-compute global vlims so colours are consistent across frames
    vlims = {}
    for key, _, cmap in fields:
        symmetric = 'RdBu' in cmap
        if key == 'v':
            all_vmag = np.sqrt(data['vx']**2 + data['vy']**2)
            vlims['v'] = (0, float(all_vmag.max()))
        elif key == 'Q':
            vlims['Q'] = (0, float(max(np.abs(data['Q']).max(), 1e-6)))
        else:
            arr = data.get(key)
            if arr is not None and arr.ndim == 3:
                if symmetric:
                    vm = float(max(np.abs(arr).max(), 1e-6))
                    vlims[key] = (-vm, vm)
                else:
                    vlims[key] = (0, float(max(arr.max(), 1e-6)))

    # Pre-compute arrow index arrays once (same for every frame)
    ix, iy = None, None
    if 'vx' in data:
        nx_size, ny_size = data['vx'].shape[1], data['vx'].shape[2]
        ix = _arrow_idx(nx_size, n_arrows)
        iy = _arrow_idx(ny_size, n_arrows)
    elif 'Q' in data and data['Q'].ndim == 3:
        nx_size, ny_size = data['Q'].shape[1], data['Q'].shape[2]
        ix = _arrow_idx(nx_size, n_arrows)
        iy = _arrow_idx(ny_size, n_arrows)

    # --- Build figure once from frame 0, collecting artist handles ----------
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.2),
                             sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    ims, qvs = [], []
    for ax, (key, label, cmap) in zip(axes, fields):
        im, qv = _draw_panel(ax, key, label, cmap, data, 0,
                             extent, Xc, Yc, n_arrows, fontsize,
                             vlims=vlims, ix=ix, iy=iy)
        ims.append(im)
        qvs.append(qv)

    title_obj = (fig.suptitle(title_fmt.format(t=0, i=0), y=1.02)
                 if title_fmt else None)
    fig.subplots_adjust(wspace=0.02, left=0.02, right=0.98, bottom=0.12, top=0.95)

    # --- Update function: swap data in-place, never touch colorbars ---------
    def _get(k, t):
        arr = data[k]
        return arr[t] if arr.ndim == 3 else arr

    def _update(frame_idx):
        for (key, _, _), im, qv in zip(fields, ims, qvs):
            if key == 'v':
                vx = _get('vx', frame_idx)
                vy = _get('vy', frame_idx)
                vmag = np.sqrt(vx**2 + vy**2) + 1e-10
                im.set_data(vmag.T)
                im.set_clim(*vlims['v'])
                if qv is not None and ix is not None:
                    qv.set_UVC((vx / vmag)[np.ix_(ix, iy)].T,
                               (vy / vmag)[np.ix_(ix, iy)].T)
            elif key == 'Q':
                Q_f = _get('Q', frame_idx)
                im.set_data((-Q_f).T)
                vmax_q = vlims['Q'][1]
                im.set_clim(-vmax_q, vmax_q)
                if qv is not None and 'q' in data and ix is not None:
                    q_f = _get('q', frame_idx)
                    nx_vec, ny_vec = nematic_to_vector(Q_f, q_f)
                    nmag = np.sqrt(nx_vec**2 + ny_vec**2) + 1e-10
                    qv.set_UVC((nx_vec / nmag)[np.ix_(ix, iy)].T,
                               (ny_vec / nmag)[np.ix_(ix, iy)].T)
            else:
                im.set_data(_get(key, frame_idx).T)
                if key in vlims:
                    im.set_clim(*vlims[key])

        if title_obj is not None:
            title_obj.set_text(title_fmt.format(t=frame_idx * dt,
                                                i=frame_idx))
        return ims

    ani = animation.FuncAnimation(fig, _update, frames=N, blit=False)
    if filename.endswith('.gif'):
        ani.save(filename, fps=fps, dpi=dpi, writer='pillow')
    else:
        ani.save(filename, dpi=dpi,
                 writer=animation.FFMpegWriter(fps=fps, codec='h264'))
    plt.close(fig)
    print(f'Saved {filename}  ({N} frames, {fps} fps)')


# Keep old name as an alias for backwards compatibility
animate_2d_gif = animate_2d