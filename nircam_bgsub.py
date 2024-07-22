import numpy as np
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage
import lmfit
from astropy.io import fits
from copy import copy
import os
import webbpsf_ext


def background_subtract_nircam_data(Database, concat, bgmodel_dir, subdir='bgsub', 
                                    fourier_shifted_without_padding=True, 
                                    nan_wrapped_data=False,
                                    fit_global_offset=True, 
                                    include_stellar_psf_component=True, 
                                    mask_snr_threshold=2, r_excl=100, q=5, 
                                    generate_plot=True, save_model=False, 
                                    fixed_bg_flux=None, use_jbt_background=False):
    """
    Perform background subtraction on NIRCam data.

    Parameters:
    
    - Database: The SpaceKLIP database object containing the aligned NIRCam
      observations.

    - concat: The concatenation on which to perform background subtraction.

    - bgmodel_dir: The directory containing the nominal background model FITS
      files.

    - subdir: The subdirectory to save the output files. Default is 'bgsub'.

    - fourier_shifted_without_padding: Whether input data were aligned using a
      Fourier shift without padding first (such that values wrapped at the
      edges). Default is True.

    - nan_wrapped_data: Whether to set wrapped pixels to NaN in the output.

    - fit_global_offset: if True, a uniform background offset is fit along with
      the background model. This corrects for offsets induced by ramp fitting
      or use of the median subtraction step in SpaceKLIP (typically for SW data
      or LW data with significant extended emission). Default is True.

    - include_stellar_psf_component: Whether to include a stellar PSF component
      when fitting the BG model to avoid oversubtracting the background (this
      component will not be subtracted from the final output file). Default is
      True.

    - mask_snr_threshold: The threshold for masking high SNR features relative
      to the estimated background SNR. E.g., a value of 5 would mask any
      features at least 5 sigma above the background level. Default is 2.

    - r_excl: The radius about the target star in pixels to exclude from
      optimization. Default is 100.

    - q: After computing BG model residuals, exclude q% of pixels from both
      ends of the residual distribution before computing chisq. This is
      intended to avoid over-/under-estimation of the background due to
      unmasked sources or artifacts. Default is 5.

    - generate_plot: Whether to generate a plot of the data, model, and
      residuals. Default is True.

    - save_model: Whether to save the background model as a separate FITS file
      in the output directory. Default is False.

    - fixed_bg_flux: if not None, the float value at which to fix the
      background flux for all files in the concatenation. Should be used only
      if fit_global_offsets is True and where the global offset and BG flux
      parameters are degenerate (e.g., data without FOV coverage of the ND
      squares). Default is None.

    - use_jbt_background: Whether to fix the background flux to the estimate
      from the JWST Backgrounds Tool. Supercedes fixed_bg_flux if both are set.
      Should be used only in the same scenarios as fixed_bg_flux, and should be
      preferred if the JWST backgrounds tools to be installed. Default is
      False.

    Returns: - Database: The updated SpaceKLIP database object.
    """
        
    def background_objective(p, im, bg0, psf0, mask, q=5):
        """
        The function producing residuals to be minimized by our optimizer.
        Note: lmfit will square these residuals and then sum them to produce a
        simple chi-squared metric.
        """
        res = (im - p['fbg']*bg0 - p['bg_offset'] - p['fpsf']*psf0)[mask]
        low, upp = np.nanpercentile(res, [q, 100.-q])
        return np.abs(res[(res >= low) & (res <= upp)])

    output_dir = os.path.join(Database.output_dir, subdir+'/')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    db_tab = Database.obs[concat]
    bgfile = f'{bgmodel_dir}{concat}_background0.fits'
    psffile = f'{bgmodel_dir}{concat}_psf0.fits'

    # Load the oversampled and extended-FOV nominal BG model
    bg0_osamp, hbg0 = fits.getdata(bgfile, ext=2, header=True)
    c_coron_bg0 = np.array([hbg0['CRPIX1'], hbg0['CRPIX2']])-1
    osamp = hbg0['OSAMP']
    
    # Optionally load the nominal stellar PSF component model
    if include_stellar_psf_component:
        psf0_osamp, hpsf0 = fits.getdata(psffile, ext=2, header=True)
        c_psf0_osamp = np.array([hpsf0['CRPIX1'], hpsf0['CRPIX2']])-1

    # Apply any blurring used for the data to the nominal model components:
    if not np.isnan(db_tab['BLURFWHM'][0]):
        blur_sigma = db_tab['BLURFWHM'][0]/np.sqrt(8.*np.log(2.))
        bg0_osamp = ndimage.gaussian_filter(bg0_osamp, blur_sigma*osamp)
        if include_stellar_psf_component:
            psf0_osamp = ndimage.gaussian_filter(psf0_osamp, blur_sigma*osamp)

    files = db_tab['FITSFILE']

    c_star = np.array([db_tab['CRPIX1'][0], db_tab['CRPIX2'][0]])-1

    h1 = fits.getheader(files[0], ext=1)
    ny, nx = h1['NAXIS2'], h1['NAXIS1']
    shift_mask0 = np.ones((ny,nx), dtype=np.float32)

    rmap = dist_to_pt(c_star, nx, ny)

    # The star should be in same position for all files, so we just do this once outside of the loop
    c_star_osamp = c_to_c_osamp(c_star, osamp)
    if include_stellar_psf_component:
        psf0_osamp_crop = pad_or_crop_about_pos(psf0_osamp, c_psf0_osamp, new_size=[ny*osamp, nx*osamp], new_cent=c_star_osamp, cval=0)
        psf0_crop = webbpsf_ext.image_manip.frebin(psf0_osamp_crop, scale=1/osamp, total=False)
    else:
        psf0_crop = 0
    
    for i, f in enumerate(files):
        p = lmfit.Parameters()
        if use_jbt_background:
            entry = db_tab[i]
            try:
                fbg0 = get_jbt_background_est(entry['EXPSTART'], entry['TARG_RA'], entry['TARG_DEC'], entry['CWAVEL'])
            except ModuleNotFoundError:
                raise ModuleNotFoundError("""
                                          JBT background estimation requires
                                          the jwst_backgrounds package. Either
                                          install jwst_backgrounds or set
                                          use_jbt_background=False and rerun.
                                          """)
            p.add('fbg', value=fbg0, vary=False)
        elif fixed_bg_flux is not None:
            p.add('fbg', value=fixed_bg_flux, vary=False)
        else:
            p.add('fbg', value=1, min=0)
        p.add('bg_offset', value=0, vary=fit_global_offset, max=0)
        p.add('fpsf', value=1, vary=include_stellar_psf_component, min=0)

        # Assume alignment+background differences between integrations are negligible so we can use the higher SNR coadded exposure
        with fits.open(f) as hdul:
            ints = hdul['SCI'].data
            errs = hdul['ERR'].data
            h1 = hdul['SCI'].header
            mask_offset = np.nanmedian(hdul['MASKOFFS'].data, axis=0)
            imshift = np.nanmedian(hdul['IMSHIFTS'].data, axis=0)

        if np.ndim(ints) == 3:
            med, err = median_combine(ints, errs)
        else: # In case using coadded data saved with only two dims
            med, err = ints, errs

        c_coron = c_star - mask_offset # post-alignment mask center position
        c_coron_osamp = c_to_c_osamp(c_coron, osamp) # mask center position in oversampled coordinates

        bg0_osamp_crop = pad_or_crop_about_pos(bg0_osamp, c_coron_bg0, new_size=[ny*osamp, nx*osamp], new_cent=c_coron_osamp, cval=1) # Crop to the data FOV
        bg0_crop = webbpsf_ext.image_manip.frebin(bg0_osamp_crop, scale=1/osamp, total=False) # Downsample to detector resolution

        if fourier_shifted_without_padding:
            shift = (imshift+np.sign(imshift))[::-1] # np.sign is just effectively adding 1 pixel to be safe in excluding wrapped data
            shift_mask = ndimage.shift(shift_mask0, shift, order=5, cval=np.nan) == 1
            # Note: we could also shift the BG model with wrapping instead. But
            # we're not actually using the wrapped pixels for any analysis
            # (hopefully)
        else:
            shift_mask = shift_mask0

        mask = (rmap > r_excl) & shift_mask
        
        snr = med/err # SNR estimate using FITS ERR extension 
        med_snr = np.nanmedian(snr[mask]) # Median SNR in the nominal background area
        low_snr = snr <= (med_snr+mask_snr_threshold) # High SNR features are those more than mask_snr_threshold sigma above the approximate BG SNR
        mask = mask & low_snr

        # Optimize the background model:
        result = lmfit.minimize(background_objective, p, args=[med, bg0_crop, psf0_crop, mask], kws=dict(q=q), method='powell')

        pfin = result.params

        # Compute the final background model and PSF component:
        bg = pfin['fbg']*bg0_crop + pfin['bg_offset']
        psf = psf0_crop*pfin['fpsf']

        f_out = output_dir+os.path.basename(os.path.normpath(f))

        with fits.open(f) as hdul:
            hdul[1].data -= bg # Subtract the BG model from the original file
            if fourier_shifted_without_padding and nan_wrapped_data:
                hdul[1].data[:, ~shift_mask] = np.nan
            hdul.writeto(f_out, overwrite=True) # Save to disk

            if save_model:
                f_model = output_dir+os.path.basename(os.path.normpath(f)).replace('.fits', '_background_model.fits')
                hdu1 = fits.ImageHDU(bg, name='BG')
                hdul_model = fits.HDUList([hdul[0], hdu1])
                if include_stellar_psf_component:
                    hdul_model.append(fits.ImageHDU(psf, name='STELLAR_PSF'))

                # Add fit params to header
                hdul_model[0].header.update(pfin.valuesdict()) 

                # Add all relevant settings to the header
                hdul_model[0].header.update(dict(fourier_shifted_without_padding = fourier_shifted_without_padding,
                                                 spaceklip_median_subtraction_applied = fit_global_offset,
                                                 mask_snr_threshold = mask_snr_threshold, r_excl = r_excl, q = q,
                                                 include_stellar_psf_component = include_stellar_psf_component))

                hdul_model.writeto(f_model, overwrite=True)
                hdul_model.close()

        if generate_plot:
            res = med - bg
            res_psfsub = res - psf
            low, upp = np.nanpercentile((res_psfsub)[mask], [q, 100.-q])
            res_inliers = np.where((res_psfsub >= low) & (res_psfsub <= upp) & mask, res_psfsub, np.nan)

            cmap = copy(mpl.cm.get_cmap('RdBu_r'))
            cmap.set_bad('white')
            clim = percentile_clim(med, 90)
            fig,axes,cbar = quick_implot(np.where(shift_mask, [med, bg+psf, res_inliers, med-bg], np.nan),
                                         cmap=cmap, clim=clim, interpolation='None', show=False, panelsize=(4,3.5),
                                         cbar=True, cbar_label='[MJy / Sr]', cbar_kwargs=dict(pad=0.008))

            labels = ['Data', 'Model (BG + PSF)', 'Masked Residuals', 'Data (BG-subtracted)']
            for j,ax in enumerate(axes):
                ax.set_title(labels[j], pad=10)

            plt.savefig(output_dir+os.path.basename(os.path.normpath(f)).replace('.fits', '_background_model.pdf'), bbox_inches='tight')
            plt.show()
        Database.update_obs(concat, i, f_out)

    return Database


def get_jbt_background_est(t, ra, dec, wavelength):
    from jwst_backgrounds import jbt
    from astropy.time import Time
    tobs = Time(t, format='mjd')
    bkg = jbt.background(ra, dec, wavelength)
    calendar = bkg.bkg_data['calendar']
    tobs0 = Time(f'{tobs.datetime.year}-01-01T00:00:00')
    thisday = int(np.round((tobs.mjd-tobs0.mjd)+1))
    Fbg = bkg.bathtub['total_thiswave'][np.where(thisday == calendar)[0][0]]
    return Fbg


def pad_or_crop_about_pos(im, pos, new_size, new_cent=None, cval=np.nan, order=5, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    ny_new, nx_new = new_size
    if new_cent is None:
        new_cent = (np.array([nx_new,ny_new])-1.)/2.
        
    nd = np.ndim(im)
    xg, yg = np.meshgrid(np.arange(nx_new, dtype=np.float32), np.arange(ny_new, dtype=np.float32))
    
    xg -= (new_cent[0]-pos[0])
    yg -= (new_cent[1]-pos[1])
    if nd == 2:
        im_out = ndimage.map_coordinates(im, np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_out = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_out[i] = ndimage.map_coordinates(im_reshaped[i], np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_out = im_out.reshape((*im.shape[:-2], ny, nx))
    return im_out


def c_to_c_osamp(c, osamp):
    return np.asarray(c)*osamp + 0.5*(osamp-1)


def dist_to_pt(pt, nx=201, ny=201, dtype=np.float32):
    """
    Returns a distance array of size (ny,nx), 
    where each pixel corresponds to the euclidean distance of that pixel from pt.
    """
    xaxis = np.arange(0, nx, dtype=dtype)-pt[0]
    yaxis = np.arange(0, ny, dtype=dtype)-pt[1]
    return np.sqrt(xaxis**2 + yaxis[:, np.newaxis]**2)


def percentile_clim(im, percentile):
    """
    Compute the color stretch limits for an image based on percentiles.

    Parameters:
    ----------
    im : array-like
        The input image.
    percentile : float or list of float
        The percentile(s) to use for computing the color stretch limits. If a single value is provided, a symmetric
        color stretch is generated spanning plus and minus the P-percentile of the absolute value of im. If two values
        are provided, they are used as the lower and upper limit percentiles.

    Returns:
    -------
    clim : array
        The lower and upper limits of the color stretch.
    """
    vals = np.unique(im)
    if np.isscalar(percentile) or len(percentile) == 1:
        clim = np.array([-1,1])*np.nanpercentile(np.abs(vals), percentile)
    else:
        clim = np.nanpercentile(vals, percentile)
    return clim


def quick_implot(im, clim=None, clim_perc=[1.0, 99.0], cmap=None,
                 show_ticks=False, lims=None, ylims=None,
                 norm=mpl.colors.Normalize, norm_kwargs={},
                 figsize=None, panelsize=[5,5], fig_and_ax=None, extent=None,
                 show=True, tight_layout=True, alpha=1.0,
                 cbar=False, cbar_orientation='vertical',
                 cbar_kwargs={}, cbar_label=None, cbar_label_kwargs={},
                 interpolation=None, sharex=True, sharey=True,
                 save_name=None, save_kwargs={}):
    """
    Plot an image or set of images with customizable options.

    Parameters:
    ----------
    im : array-like
        The input image(s) to plot. If im is a 2D array, a single panel will be created. If im is a 3D array, a row of
        panels will be created. If im is a 4D array, a grid of panels will be created. E.g., 
        im=[[im1, im2], [im3, im4], [im5, im6]] will create a plot with 3 rows and 2 columns.
    clim : str or tuple, optional
        The color stretch limits. If a string is provided, it should contain a comma-separated pair of values.
        If a tuple is provided, it should contain the lower and upper limits of the color stretch.
    clim_perc : float or list of float, optional
        The percentile(s) to use for computing the color stretch limits. If a single value is provided, a symmetric
        color stretch is generated spanning plus and minus the P-percentile of the absolute value of im. If two values
        are provided, they are used as the lower and upper limit percentiles.
    cmap : str or colormap, optional
        The colormap to use for the image.
    show_ticks : bool, optional
        Whether to show ticks on the plot.
    lims : tuple, optional
        The x-axis (and y-axis if ylims is not provided) limits of the plot.
    ylims : tuple, optional
        The y-axis limits of the plot.
    norm : matplotlib.colors.Normalize or subclass, optional
        The normalization class to use for the color mapping.
    norm_kwargs : dict, optional
        Additional keyword arguments to pass to the normalization class.
    figsize : tuple, optional
        The size of the figure in inches. If not provided, the size will be determined based on the number of panels and
        the panelsize argument.
    panelsize : list, optional
        The size of each panel in the figure. 
    fig_and_ax : tuple, optional
        A tuple containing a matplotlib Figure and Axes object to use for the plot.
    extent : array-like, optional
        The extent of the plot as [xmin, xmax, ymin, ymax].
    show : bool, optional
        Whether to show the plot or return the relevant matplotlib objects.
    tight_layout : bool, optional
        Whether to use tight layout for the plot.
    alpha : float, optional
        The transparency of the image.
    cbar : bool, optional
        Whether to show a colorbar.
    cbar_orientation : str, optional
        The orientation of the colorbar ('vertical' or 'horizontal').
    cbar_kwargs : dict, optional
        Additional keyword arguments to pass to the colorbar.
    cbar_label : str, optional
        The label for the colorbar.
    cbar_label_kwargs : dict, optional
        Additional keyword arguments to pass to the colorbar label.
    interpolation : str, optional
        The interpolation method to use with imshow.
    sharex : bool, optional
        Whether to share the x-axis among subplots.
    sharey : bool, optional
        Whether to share the y-axis among subplots.
    save_name : str, optional
        The filename to save the plot. Plot will be saved only if this argument is provided.
    save_kwargs : dict, optional
        Additional keyword arguments to pass to the save function.

    Returns:
    -------
    fig : matplotlib Figure
        The created Figure object.
    ax : matplotlib Axes or array of Axes
        The created Axes object(s).
    cbar : matplotlib Colorbar, optional
        The created Colorbar object.

    Notes:
    ------
    - If clim is a string, it should contain a comma-separated pair of values. The values can be interpretable as floats,
      in which case they serve as the corresponding entry in the utilized clim. Alternatively, they can contain a '%'
      symbol, in which case they are used as a percentile bound. For example, clim='0, 99.9%' will yield an image with a
      color stretch spanning [0, np.nanpercentile(im, 99.9)]. If clim contains a '*', the options can be multiplied.
    - If clim is not provided, the color stretch limits will be computed based on the clim_perc parameter.
    - If clim_perc is a single value, the lower and upper limits of the color stretch will be symmetric.
    - If clim_perc is a list of two values, they will be used as the lower and upper limit percentiles.
    """
    if isinstance(clim, str):
        s_clim = [i.strip() for i in clim.split(',')]
        clim = []
        for s in s_clim:
            if s.isdigit():
                clim.append(float(s))
            elif '%' in s:
                if '*' in s:
                    svals = []
                    for si in s.split('*'):
                        if '%' in si:
                            svals.append(np.nanpercentile(im, float(si.replace('%',''))))
                        else:
                            svals.append(float(si))
                    clim.append(np.prod(svals))
                else:
                    clim.append(np.nanpercentile(im, float(s.replace('%',''))))
            else:
                raise ValueError(
                    """
                    If clim is a string, it should contain a comma separating
                    two entries. These entries should be one of:
                    a) interpretable as a float, in which case they serve as the 
                    corresponding entry in the utilized clim, b) they should contain a
                    % symbol, in which case they are used as a percentile bound;
                    e.g., clim='0, 99.9%' will yield an image with a color
                    stretch spanning [0, np.nanpercentile(im, 99.9)], or c) they
                    should contain a '*' symbol, separating either of the 
                    aforementioned options, in which case they will be multiplied.
                    """)
            
    elif clim is None:
        clim = percentile_clim(im, clim_perc)
        
    if ylims is None:
        ylims = lims
        
    normalization = norm(vmin=clim[0], vmax=clim[1], **norm_kwargs)

    if np.ndim(im) in [2,3,4]:
        im_4d = np.expand_dims(im, np.arange(4-np.ndim(im)).tolist()) # Expand dimensions to 4D if not already to easily extract nrows and ncols
        nrows, ncols = im_4d.shape[0:2]
    else:
        raise ValueError("Argument 'im' must be a 2, 3, or 4 dimensional array")
    n_ims = nrows * ncols

    if fig_and_ax is None:
        if figsize is None:
            figsize = np.array([ncols,nrows])*np.asarray(panelsize)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    else:
        fig, ax = fig_and_ax

    if n_ims == 1:
        ax, im = [ax], [im.squeeze()]
    else:
        im = np.asarray(im).reshape((nrows*ncols, *np.shape(im)[-2:]))
        ax = np.asarray(ax).flatten()

    for ax_i, im_i in zip(ax, im):
        implot = ax_i.imshow(im_i, origin='lower', cmap=cmap, norm=normalization, extent=extent, alpha=alpha, interpolation=interpolation)
        if not show_ticks:
            ax_i.set(xticks=[], yticks=[])
        ax_i.set(xlim=lims, ylim=ylims)
    if tight_layout:
        fig.tight_layout()
    if cbar:
        cbar = fig.colorbar(implot, ax=ax, orientation=cbar_orientation, **cbar_kwargs)
        cbar.set_label(cbar_label, **cbar_label_kwargs)
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', **save_kwargs)
    if show:
        plt.show()
        return None
    if n_ims == 1:
        ax = ax[0]
    if cbar:
        return fig, ax, cbar
    return fig, ax


def median_combine(imcube, errcube=None):
    im = np.nanmedian(imcube, axis=0)
    if errcube is None:
        return im,None 
    n = np.sum(np.isfinite(imcube), axis=0)
    sig_mean = np.sqrt(np.nansum(errcube**2, axis=0))/n
    err = np.sqrt(np.pi*(2*n+1)/(4*n)) * sig_mean
    return im,err