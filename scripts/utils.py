 
'''
Functions used throughout the analyses, inludes:
retrieving gene expression and filtering by gene set
plotting on the brain
spin tests on correlations
class distribution

Author: Moohebat
'''

import abagen
import colormaps as cmaps
from brainspace.utils.parcellation import map_to_labels
from enigmatoolbox.utils.parcellation import parcel_to_surface
import matplotlib.pyplot as plt
from netneurotools import datasets
from netneurotools.freesurfer import parcels_to_vertices
from neuromaps.datasets import fetch_fsaverage
from nibabel import freesurfer
from nilearn.datasets import fetch_atlas_schaefer_2018
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, zscore
import seaborn as sns
from surfplot import Plot

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({"font.size": 8})

path_data = './data/'


def load_expression(scale):

    '''get expression dictionary from abagen'''

    schaefer = fetch_atlas_schaefer_2018(n_rois=scale) # 7networks and 1mm res.

    expression = abagen.get_expression_data(schaefer['maps'],
                                            lr_mirror='bidirectional', 
                                            missing='interpolate', 
                                            return_donors=True)
    return expression



def filter_expression_ds(expression, ds):
    '''
    filter expression matrix based on given differential stability value
    and average across donors
    '''
    
    # keeping stable genes across donors and samples.
    rexpression, diff_stability = abagen.correct.keep_stable_genes(list(expression.values()), 
                                                           threshold=ds, 
                                                           percentile=False, 
                                                           return_stability=True)
    # rexpression is a type list of dataframes
    # rexpression is still by donor, so we are avergaing over donors
    expression_ds = pd.concat(rexpression).groupby('label').mean()

    return expression_ds, diff_stability



def geneset_expression(expression, gene_list, filename, outpath, save=False):

    '''
    filters ahba expression data for a given gene set
    '''
    # getting the expression values for a given gene list
    filtered_expression = expression[expression.columns.intersection(gene_list)]
    # save
    if save:
        filtered_expression.to_csv(outpath+filename+'_exp.csv')

    return filtered_expression



def load_expression_lh(scale):

    '''
    get expression dictionary from abagen only for the left hemisphere
    '''

    schaefer = fetch_atlas_schaefer_2018(n_rois=scale) # 7networks and 1mm res.
    # getting ahba expression data
    # set lr_mirror to None
    expression = abagen.get_expression_data(schaefer['maps'],
                                            lr_mirror=None, 
                                            missing='interpolate', 
                                            return_donors=True)
    return expression



def dk_plot(data):
    '''
    plot brain map in desikian-killiany parcellation
    '''
    #loading fsaverage 10k surface
    surfaces = fetch_fsaverage(density='10k')
    lh, rh = surfaces['pial']
    #lh_sulc, rh_sulc = surfaces['sulc']
    p = Plot(surf_lh=lh, surf_rh=rh)
    surface = parcel_to_surface(data, 'aparc_fsa5')
    #color_range = (min(x), max(x))
    p.add_layer(surface, cmap='RdBu', cbar=True)
    kws = {'fontsize': 8, 'n_ticks': 2, 'shrink': 0.4, 'aspect': 10 ,'draw_border': False}
    fig = p.build(figsize=(3,3), cbar_kws= kws)
    fig.show()



def plot_schaefer_fsaverage(data, hemi=None, cmap = 'plasma', resolution=400):
    '''
    function to plot parcellated schaefer data onto fsaverage surface
    uses surfplot
    '''

    # load schaefer atlas
    if resolution == 400:
        scale = '400Parcels7Networks'
    elif resolution == 100:
        scale = '100Parcels7Networks'
    elif resolution == 600:
        scale = '600Parcels7Networks'
    
    # fetch schaefer parcellation files
    schaefer = datasets.fetch_schaefer2018('fsaverage')[str(resolution)+'Parcels7Networks']

    # convert parcellated data into vertex-wise data
    x = parcels_to_vertices(data, lhannot=schaefer.lh, rhannot=schaefer.rh)

    # load surface files
    surfaces = fetch_fsaverage(density='164k')
    # sulc_lh, sulc_rh = surfaces['sulc']
    lh, rh = surfaces['inflated']

    #keeping only the left hemisphere data
    if hemi == 'L':
        x = x[:len(x)//2]
        p = Plot(surf_lh=lh, zoom=1.2)
    
    # lh_sulc, rh_sulc = surfaces['sulc']
    p = Plot(surf_lh=lh, surf_rh=rh, zoom=1.5)
        
    # add data layer to surface    
    p.add_layer(x, cmap=cmap, cbar=True)

    kws = {'fontsize': 8, 'n_ticks': 2, 'shrink': 0.4, 'aspect': 10 ,'draw_border': False}
    fig = p.build(figsize=(3,3), cbar_kws= kws)
    plt.tight_layout()
    # fig.show()



def glasser_plot(data, hemi=None, color=cmaps.matter_r, 
                 outline=True, outlinecolor='gray', 
                 roi=False, roi_drop=[b'???'], 
                 color_range=None, brightness=None,
                 views=['lateral', 'medial', 'dorsal', 'ventral', 'posterior'],
                 interactive=False):

    #loading fsaverage 164k surface
    surfaces = fetch_fsaverage(density='164k')
    lh, rh = surfaces['inflated']
    p = Plot(surf_lh=lh, surf_rh=rh, zoom=1.5)

    # convert parcellated data into vertex-wise data
    x = parcels_to_vertices(data, 
                        lhannot=path_data+'lh.HCPMMP1.annot', 
                        rhannot=path_data+'rh.HCPMMP1.annot', drop=roi_drop)     
    
    if hemi == 'L':
        x = x[:len(x)//2]
        p = Plot(surf_lh=lh, zoom=1.2, brightness=brightness,
                layout='row', views=views, size=(1200, 200))
    elif hemi == 'R':
        x = x[len(x)//2:]
        p = Plot(surf_rh=rh, zoom=1.2, brightness=brightness,
                layout='row', views=views, size=(1200, 200))

    p.add_layer(x, cmap=color, cbar=True, color_range=color_range)
    p.add_layer(x, as_outline=outline, cbar=False, cmap=outlinecolor)

    kws = {'fontsize': 8, 'n_ticks': 2, 'shrink': 0.4, 
           'aspect': 10, 'draw_border': False}
    fig = p.build(cbar_kws= kws)

    # p.show() gives you the interactive mode
    if interactive:
        p.show()



def glasser_plot_roi(data, hemi=None, color=cmaps.matter_r, 
                 outline=True, outlinecolor='gray', 
                 roi=False, roi_idx=None, 
                 color_range=None, brightness=None,
                 views=['lateral', 'medial', 'dorsal', 'ventral', 'posterior'],
                 interactive=False,):

    # loading glasser 360 atlas annot files for plotting
    glasser_lh = freesurfer.read_annot(path_data+'atlases/lh.HCPMMP1.annot')
    glasser_rh = freesurfer.read_annot(path_data+'atlases/rh.HCPMMP1.annot')

    # look into atlas shape
    labels_lh, ctab_lh, names_lh = glasser_lh
    labels_rh, ctab_rh, names_rh = glasser_rh
    
    #getting the labels for the whole brain, 163842*2 vertices
    labeling = np.concatenate((labels_lh, labels_rh))
    #shape (327684,), labels from 0 to 180.
    # 
    if roi:
        map = map_to_labels(data, labeling, 
                            mask= np.isin(labeling, roi_idx))
    else:
        map = map_to_labels(data, labeling, mask=labeling != 0)

    #loading fsaverage 164k surface
    surfaces = fetch_fsaverage(density='164k')
    lh, rh = surfaces['inflated']

    # just plotting the left hemisphere
    if hemi == 'L':
        p = Plot(surf_lh=lh, zoom=1.7, brightness=brightness,
                layout='column', views=views, size=(400, 400))

        p.add_layer(map[:163842], cmap=color, cbar=False, color_range=color_range)
        p.add_layer(map[:163842], as_outline=outline, cbar=False, cmap=outlinecolor)
    # both hemisphere
    else:
        p = Plot(surf_lh=lh, surf_rh=rh, zoom=1.2, brightness=brightness)
        p.add_layer(map, cmap=color, cbar=True, color_range=color_range)
        p.add_layer(map, as_outline=outline, cbar=False, cmap=outlinecolor)

    kws = {'fontsize': 8, 'n_ticks': 2, 'shrink': 0.4, 
           'aspect': 10, 'draw_border': False}
    fig = p.build(cbar_kws= kws)

    # p.show() gives you the interactive mode
    if interactive:
        p.show()

    return fig, map



def corr_spin_test(data, map, spins, 
                   scattercolor='sandybrown', 
                   linecolor='grey', 
                   plot=False):
    '''
    calculate correlation between two brain maps
    and calculate the p-spin'
    '''

    # generating spins and calculating null correlation distribution
    nspins = spins.shape[1]
    corr_null = np.zeros((spins.shape[1],))
    corr, _ = spearmanr(data, map)
    for i in range(spins.shape[1]):
        corr_null[i], _ = spearmanr(data[spins[:, i]], map)

    # calculating p_spin
    p_spin = (
        1
        + np.sum(abs(corr_null - np.mean(corr_null)) >= abs(corr - np.mean(corr_null)))
    ) / (nspins + 1)

    # plotting correlation plot and reporting pval
    if plot:
        plt.figure(figsize=(3, 3))
        sns.regplot(x=data,
            y=map,
            scatter=True,
            fit_reg=True,
            color=scattercolor,
            ci=None,
            scatter_kws={'linewidth': 0, 's': 12, 'alpha': 1},
            line_kws={'color': linecolor, 'lw': 0.7, 'alpha': 0.8},)
        sns.despine()
        plt.title('r ={:.2f}'.format(corr) + ',  p_spin ={:.2f}'.format(p_spin))
        plt.tight_layout()
        # plt.show()
    return corr, corr_null, p_spin




def pair_corr_spin(x, y, spins):
    '''
    calculate pairwise correlation between 
    two dataframes with spin test
    '''

    corr_df = pd.DataFrame(columns=x.columns, index=y.columns, dtype=np.float64)
    pspin_df = pd.DataFrame(columns=x.columns, index=y.columns, dtype=np.float64)
    for col in x.columns:
        for col2 in y.columns:
            corr_df.loc[col2, col], _, pspin_df.loc[col2, col] = corr_spin_test(np.array(x[col]),
                                                                                      y[col2], 
                                                                                      spins,
                                                                                      plot=False)
    return corr_df, pspin_df



def plot_heatmap(corr_df, pspin_df, 
                 linecolor='white', linewidths=0.3,
                 asteriks=True, edge=False,
                 annot=False):
    '''
    plot correlation heatmap with significance
    '''
    import colormaps as cmaps
    ax = sns.heatmap(corr_df.T, annot=annot, 
                     cmap=cmaps.BlueWhiteOrangeRed, 
                     vmin=-1, vmax=1,
                     cbar_kws={'shrink': 0.25, 
                               'ticks': [-1.00, 0.0, 1.00], 
                               'pad': 0.02},
                    linewidths=linewidths, linecolor=linecolor,
                    fmt='.2f')
    if asteriks:
    # asteriks for significant ones
        for i in range(pspin_df.shape[0]):
            for j in range(pspin_df.shape[1]):
                if pspin_df.iloc[i, j] <= 0.0001:
                    ax.text(i+0.6, j+0.4, '****', ha='left', va='bottom', 
                            color='k')
                elif pspin_df.iloc[i, j] < 0.001:
                    ax.text(i+0.6, j+0.4, '***', ha='left', va='bottom', 
                            color='k')
                elif pspin_df.iloc[i, j] < 0.01:
                    ax.text(i+0.6, j+0.4, '**', ha='left', va='bottom', 
                            color='k')
                elif pspin_df.iloc[i, j] < 0.05:
                    ax.text(i+0.6, j+0.4, '*', ha='left', va='bottom', 
                            color='k')
    if edge:
    # bold edge for significant ones
        for i in range(pspin_df.shape[0]):
            for j in range(pspin_df.shape[1]):
                if pspin_df.iloc[i, j] < 0.05:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, 
                                            fill=False, 
                                            edgecolor='black', 
                                            lw=0.7))
    for spine_name, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        
    plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.show()



def class_enrichment(map, class_labels, spins, order, outpath, filename):
    '''
    function to calculate class distribution of a given brain map
    '''
    nets = order
    num_nets = len(nets)
    map = zscore(map)

    fig, axs = plt.subplots(1, num_nets, 
                            figsize=(1.2*num_nets, 2),
                            sharey=True,
                            sharex=True)

    for ax, net in zip(axs, nets):
        emp_enrich = {}
        null_enrich = {}
        pval_enrich = {}
        for key in map.columns:
            mask = (class_labels == net)
            #average expression in network
            emp_enrich[key] = np.mean(map[key][mask])

            nulls = np.zeros([spins.shape[1],])
            for i in range(spins.shape[1]):
                data_null = np.array(map[key][spins[:, i]])
                nulls[i]= np.mean(data_null[mask])
            #spin null distribution for network
            null_enrich[key] = np.array(nulls)
            #calculate spin pvalue
            pval_enrich[key] = (1 + np.sum(np.abs((nulls - np.mean(nulls)))
                            >= abs((emp_enrich[key] - np.mean(nulls))))) / (spins.shape[1] + 1)
        print(pval_enrich)

        #bar plot
        colors = ['coral' if pval < 0.05 else 'lightgrey' for pval in pval_enrich.values()]
        ax.bar(emp_enrich.keys(), 
                emp_enrich.values(),
                color=colors,
                width=0.7)

        ax.set_title(net)
        ax.set_ylabel('zscore(mean expression)')
        ax.set_xlabel('')
        ax.set_xticklabels(emp_enrich.keys(), 
                        rotation = 45, 
                        ha='right', 
                        rotation_mode='anchor')
        sns.despine()

    for i, ax in enumerate(axs):
        if i != 0:
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
    plt.ylim(-2,2)
    plt.tight_layout()
    plt.savefig(outpath+filename+'.svg')
    plt.show()