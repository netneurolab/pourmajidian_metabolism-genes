

'''
Lifespan trajectories of energy pathway expressions
1. clean up Brainspan RNA-seq data
2. plot trajectories
3. simple stats for samples
Author: Moohebat
Date: 21/12/2024

'''

# import packages
import pickle
import qnorm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.lib import grdevices
from rpy2.robjects import pandas2ri
pandas2ri.activate()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'
path_bs = path_data+'brainspan/genes_matrix_csv/'

# load brainspan data
# data downloaded from brainspan 'RNA-Seq Gencode v10 summarized to genes'
# sample metadata
sample_data = pd.read_csv(path_bs + 'columns_metadata.csv')
sample_data = sample_data.drop("column_num", axis=1)
sample_data.index = sample_data.index+1
# 524 samples

# stats
np.unique(sample_data['donor_id']).shape #42 donors
np.unique(sample_data['age']).shape #31 developmental stages
np.unique(sample_data['structure_acronym']).shape #26 structure acronyms

# creating new sample attributes
# age order
order = ['8 pcw', '9 pcw', '12 pcw', '13 pcw', '16 pcw', 
         '17 pcw', '19 pcw', '21 pcw','24 pcw', '25 pcw', 
         '26 pcw', '35 pcw', '37 pcw','4 mos','10 mos',
         '1 yrs', '2 yrs', '3 yrs','4 yrs','8 yrs',
         '11 yrs', '13 yrs', '15 yrs', '18 yrs', '19 yrs',
         '21 yrs', '23 yrs','30 yrs', '36 yrs', '37 yrs',
         '40 yrs']

# adding attributes: age category, cortex and network subdivisions
def map_to_category(list_names, list_items):
    map_to_category = {item: category for category, items in \
                       zip(list_names, list_items) for item in items}
    return map_to_category

# broad age category Li et al 2018
fetal = ['8 pcw', '9 pcw', '12 pcw', '13 pcw', '16 pcw', 
         '17 pcw', '19 pcw', '21 pcw','24 pcw', '25 pcw', 
         '26 pcw', '35 pcw', '37 pcw']
infant = ['4 mos','10 mos', '1 yrs']
child = ['2 yrs', '3 yrs','4 yrs','8 yrs']
adolescent = ['11 yrs', '13 yrs', '15 yrs', '18 yrs', '19 yrs']
adult = ['21 yrs', '23 yrs','30 yrs', '36 yrs', '37 yrs',
         '40 yrs']
# map
groups = [fetal,infant,child,adolescent,adult]
names = ['fetal','infant','child','adolescent','adult']
sample_data['age_group'] = sample_data['age'].map(map_to_category(names, groups))

# age category kang et al 2011
early_fetal = ['8 pcw', '9 pcw', '12 pcw']
mid_fetal = ['13 pcw', '16 pcw', '17 pcw', '19 pcw', '21 pcw']
late_fetal = ['24 pcw', '25 pcw', '26 pcw', '35 pcw', '37 pcw']
infancy = ['4 mos', '10 mos', '1 yrs']
early_child = ['2 yrs', '3 yrs', '4 yrs']
late_child = ['8 yrs', '11 yrs']
adolescent = ['13 yrs', '15 yrs', '18 yrs', '19 yrs']
adult = ['21 yrs', '23 yrs', '30 yrs', '36 yrs', '37 yrs', '40 yrs']
# map
names = ['early_fetal','mid_fetal','late_fetal','infancy', 
         'early_child','late_child','adolescent','adult']
group2 = [early_fetal, mid_fetal, late_fetal, infancy,
          early_child, late_child, adolescent, adult]
sample_data['age_group2'] = sample_data['age'].map(map_to_category(names, group2))

# age periods kang et al 2011
p2 = ['8 pcw', '9 pcw']
p3 = ['12 pcw',]
p4 = ['13 pcw']
p5 = ['16 pcw', '17 pcw',]
p6 = ['19 pcw', '21 pcw']
p7 = ['24 pcw','25 pcw', '26 pcw', '35 pcw','37 pcw']
p8 = ['4 mos']
p9 = ['10 mos',]
p10 = ['1 yrs', '2 yrs', '3 yrs','4 yrs',]
p11 = ['8 yrs','11 yrs',]
p12 = ['13 yrs', '15 yrs', '18 yrs', '19 yrs']
p13 = ['21 yrs', '23 yrs','30 yrs', '36 yrs', '37 yrs']
p14 = ['40 yrs']
# map
periods = [p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12, p13, p14]
names = ['p2','p3','p4','p5','p6','p7','p8','p9',
         'p10','p11','p12', 'p13', 'p14']
sample_data['period'] = sample_data['age'].map(map_to_category(names, periods))

# adding cortex, non_cortex division
regions = np.unique(sample_data['structure_name']) # 26
cortex = []
subcortex = []
for region in regions:
    if 'cortex' in region:
        cortex.append(region)
    else:
        subcortex.append(region)

cortex.remove('cerebellar cortex')
subcortex.append('cerebellar cortex')
# there are 15 cortical and 11 subcortical regions
# map
names = ['cortex', 'subcortex']
ctx = [cortex, subcortex]
sample_data['ctx'] = sample_data['structure_name'].map(map_to_category(names, ctx))

# anatomical divisions
front = ['OFC', 'DFC', 'VFC', 'MFC', 'M1C']
parietal = ['PCx', 'IPC', 'S1C',]
temp = ['STC', 'ITC', 'TCx', 'A1C']
occip = ['Ocx', 'V1C']
lim = ['AMY', 'HIP']

networks = {
    'front': front,
    'par': parietal,
    'temp': temp,
    'occ': occip,
    'lim': lim
    }

def to_networks(regions):
    for key, value in networks.items():
        if regions in value:
            return key

sample_data['network'] = sample_data['structure_acronym'].apply(to_networks)

# convert age to postconception days
sample_data['age_days'] = np.nan

for i, age in enumerate(sample_data['age']):
    if 'pcw' in age:
        sample_data.loc[i+1, 'age_days'] = float(age.split(' ')[0]) * 7
    elif 'mos' in age:
        sample_data.loc[i+1, 'age_days'] = 40 * 7 + float(age.split(' ')[0]) * 30.43
    elif 'yrs' in age:
        sample_data.loc[i+1, 'age_days'] = 40 * 7 + float(age.split(' ')[0]) * 12 * 30.43

######################
# sample data cleanup
# from now we're focusing on cortical regions
sample_data2 = sample_data[sample_data['ctx'] == 'cortex']
# sample_data2 = sample_data
# 1. keep regions with at least 1 sample in each age group
np.unique(sample_data2['structure_acronym']).shape  # 26 unique regions
groups = sample_data2.groupby(['structure_acronym', 
                              'age_group']).size().unstack(fill_value=0)

structures = groups[(groups >= 1).all(axis=1)].index
sample_data2 = sample_data2[sample_data2['structure_acronym'].isin(structures)]
dropped = list(set(sample_data['structure_acronym']) - set(sample_data2['structure_acronym']))
# dropped ['PCx', 'M1C-S1C', 'TCx', 'Ocx']
# 352 cortical samples left

####################
# expression matrix
bs_exp = pd.read_csv(path_bs + 'expression_matrix.csv', header=None) 
bs_exp = bs_exp.iloc[:, 1:] # (52376 gene x 524 sample)

# gene metadata
gene_data = pd.read_csv(path_bs + 'rows_metadata.csv', header=0, usecols=['gene_symbol'])
#52376 genes

###########
# QC
# 1. drop duplicate genes
gene_uniq = gene_data.drop_duplicates()
bs_exp = bs_exp.iloc[gene_uniq.index, :] # (47808 rows x 524 columns)

# 2. drop non-expressed genes
# keep genes that: RPKM of >=1 in 80% of all the samples from each spatiotemporal point
bs_exp = bs_exp.loc[:, sample_data2.index]
bs_exp2 = pd.concat([sample_data2[['structure_acronym', 'age_group']], bs_exp.T], axis=1)

grouped = bs_exp2.groupby(['structure_acronym', 'age_group'])

to_keep = pd.DataFrame()
for (region, age_grp), group_data in grouped:
    
    exp = group_data.drop(['structure_acronym', 'age_group'], axis=1)
    # RPKM >= 1 in >= 80 of samples for a region and a period
    to_keep[region+age_grp] = (exp >= 1).sum(axis=0) >= (0.8 * exp.shape[0])

# to_keep = to_keep[to_keep.all(axis=1)]
to_keep = to_keep[(to_keep.sum(axis=1) == len(grouped))]
gene_idx = to_keep.index
# this keeps 8370 genes

# filtered expression
bs_exp = bs_exp.loc[gene_idx, :]
gene_qc = gene_uniq.loc[gene_idx, :]

bs_df = pd.concat([gene_qc, bs_exp], axis=1) 
# 8370 genes by 352 samples

# save the cleaned expression matrix and sample data
bs_df.to_csv(path_result+'brainspan_rna_exp_qc.csv')

sample_data2.to_csv(path_result+'sample_data_rna_qc.csv')


#####################
# quartile normalize

# load brainspan
sample_data = pd.read_csv(path_result+'sample_data_rna_qc.csv').set_index('column_num')
bs_df = pd.read_csv(path_result+'brainspan_rna_exp_qc.csv').iloc[:, 1:]

# log2 transform
bs_df.iloc[:, 1:] = np.log2(bs_df.iloc[:, 1:]+1)

# normalize to 75th percentile across donors
df = pd.concat([sample_data['donor_id'].reset_index(drop=True), 
                bs_df.set_index('gene_symbol').T.reset_index(drop=True)], 
                axis=1)

# get the 75th percentile expression value for each donor
percentile_75 = df.groupby('donor_id').apply(lambda group: np.percentile(group.iloc[:, 1:].values.flatten(), 75))
# avergae across all donors
mean_p75 = np.mean(percentile_75)

df_norm = pd.DataFrame()
for donor in np.unique(sample_data['donor_id']):
    group = df.groupby('donor_id').get_group(donor)
    # get 75th percentile of expression values in each donor
    percent75 = np.percentile(group.iloc[:, 1:].values.flatten(), 75)
    # scale each donor's expression values by their 75th precentile
    # then multiply by the average 75th percentile across all donors
    group.iloc[:, 1:] = (group.iloc[:, 1:] / percent75) * mean_p75
    df_norm = pd.concat([df_norm, group], axis=0)

df_norm = df_norm.sort_index()

df_norm.to_csv(path_result+'brainspan_exp_rna_uqnorm.csv')

##########################
# energy analysis

# read brainspan normalzied data
sample_data = pd.read_csv(path_result+'sample_data_rna_qc.csv').set_index('column_num')

df_norm = pd.read_csv(path_result+'brainspan_exp_rna_uqnorm.csv').iloc[:, 1:]

final_df = df_norm.drop('donor_id', axis=1).T.reset_index(names='gene_symbol')

# loading energy gene sets for curation based on brainspan data
with open(path_result+'energy_genelist_dict.pickle', 'rb') as f:
    energy_genes = pickle.load(f)

# fix the issue with atpsynth gene symbols in brainspan
atp_genes = ['ATP5A1', 'ATP5B', 'ATP5C1', 'ATP5D',
             'ATP5E', 'ATP5I', 'ATP5J2', 'ATP5L', 
             'ATP5F1', 'ATP5H', 'ATP5J', 'ATP5O']
energy_genes['atpsynth'] = np.array(atp_genes)

# last 14 genes in oxphos are atpsynth genes
oxphos_genes = energy_genes['oxphos'][:-14]
oxphos_genes = np.append(oxphos_genes, atp_genes)
energy_genes['oxphos'] = oxphos_genes

###############################
# for ahba harmonized analysis
with open(path_result+'energy_expression_matrix.pickle', 'rb') as f:
    energy_exp = pickle.load(f)

# make ahba energy gene list
energy_genes = {}
for key, value in energy_exp.items():
    energy_genes[key] = list(value.columns)

# fix atp genes
atp_ahba = ['ATP5A1', 'ATP5B', 'ATP5I', 
            'ATP5J2', 'ATP5F1', 'ATP5J']
energy_genes['atpsynth'] = atp_ahba

# first 6 are atpsynth genes
oxphos_ahba = energy_genes['oxphos'][6:]
oxphos_ahba = np.append(oxphos_ahba, atp_ahba)
energy_genes['oxphos'] = oxphos_ahba

################################
# get energy expression and mean
bs_exp_energy = {}
bs_mean_energy = {}
for key, value in energy_genes.items():
        bs_exp_energy[key] = final_df[final_df.gene_symbol.isin(value)]
        bs_mean_energy[key] = np.mean(bs_exp_energy[key].iloc[:, 1:], axis=0)

# converting mean expression dict to dataframe
bs_mean_energy_df = (pd.DataFrame.from_dict(bs_mean_energy))
# add sample info for plotting
df = pd.concat([sample_data2, bs_mean_energy_df], axis=1)

#focusing on main energy pathways
energy_main =['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate', 'kb_util']

#############
# final plot
fig_size = {'age_group': (8, 5), 'age_group2': (8, 5)}
widths = {'age_group': 0.4, 'age_group2': 0.4}
rotations = {'age_group': 35, 'age_group2': 35}
categories = ['age_group', 'age_group2']

for category in categories:
    fig, axes = plt.subplots(2, 3, figsize=(fig_size[category]), sharex=True, sharey=False)
    axes = axes.flatten()
    
    for ax, key in zip(axes, energy_main):
        sns.boxplot(data=df,
            x=category,
            y=key,
            boxprops={'facecolor': 'none', 'edgecolor': 'darkgrey'},
            whiskerprops={'color': 'darkgrey'},
            capprops={'visible': False},
            medianprops={'color': 'crimson'},
            showfliers=False,
            linewidth=0.7,
            width=widths[category],
            ax=ax,)
        
        sns.stripplot(data=df,
            x=category,
            y=key,
            color='sandybrown',
            size=3,
            alpha=0.4,
            jitter=True,
            zorder=-5,
            ax=ax,)
        
        sns.lineplot(data=df,
            x=category,
            y=key,
            estimator=np.median,
            ci=None,
            sort=False,
            color='mediumvioletred',
            marker='o',
            markersize=0.7,
            markeredgewidth=0,
            ax=ax,
            linewidth=0.7,)
        
        ax.set_title(key)
        ax.set_xlabel(None)
        ax.set_ylabel('expression (normalized RPKM)')
        sns.despine(ax=ax)
        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotations[category], 
                        ha='right', rotation_mode='anchor')
            if ax != axes[0]:
                    ax.set_ylabel(None)
    plt.tight_layout()
    plt.savefig(path_fig + category + '_box_line_quant75.svg')
    plt.show()


#################
# smoothed curves

# add log(age) column
df['log_age_days'] = np.log10(df['age_days'].values.astype('float'))

# convert to long format for grid
df_long = pd.melt(df, id_vars=['log_age_days'], value_vars=energy_main, 
                  var_name='pathway', value_name='expression')

df_long['pathway'] = pd.Categorical(df_long['pathway'], 
                                    categories=energy_main, ordered=True)

# convert to r dataframe fro ggplot
df_r = pandas2ri.py2rpy(df_long)

# plot
grdevices.svg(path_fig+"brainspan_geom_loess_main1.svg", width=8, height=5)

pp = (ggplot2.ggplot(df_r) +
      ggplot2.aes_string(x='log_age_days', y='expression') +
      ggplot2.geom_point(color='sandybrown', size=1.7, stroke=0,
                         alpha=0.4,) +
      ggplot2.geom_smooth(method='loess', se=False, 
                          color='mediumvioletred', size=0.6) +
      ggplot2.facet_wrap("~pathway", scales='free') +
      ggplot2.theme(axis_line=ggplot2.element_line(color='black'),
                    panel_background=ggplot2.element_blank(),
                    panel_grid_major=ggplot2.element_blank(),
                    panel_grid_minor=ggplot2.element_blank()) +
                    ggplot2.geom_vline(xintercept=np.log10(40*7), 
                                       color='grey', linetype='dashed', 
                                       size=0.7, show_legend=False))
pp.plot()
grdevices.dev_off()


# stripplot version for original age stages
for category in ['age']:
    for key in energy_genes.keys():
        if key in energy_main:
            plt.figure(figsize=(4, 3))
            sns.stripplot(
                data=df,
                x=category,
                y=key,
                s=3,
                color='sandybrown',
                edgecolor='grey',
                linewidth=0.2,
                alpha=0.9,
                legend=False, 
                jitter=True,
            )
            plt.ylabel('expression (RPKM)')
            plt.title(key)
            plt.xlabel(None)
            plt.xticks(rotation=90)
            sns.despine()
            plt.tight_layout()
            # plt.savefig(path_fig+key+'_'+ category +'_strip.svg')
            plt.show()

############################################
# save summary of brainspan into latex table

num_samples = sample_data.groupby('donor_name').size()

names = ['early_fetal','mid_fetal','late_fetal','infancy', 
         'early_child','late_child','adolescent','adult']

table = sample_data.groupby('age_group2')['age'].apply(set).reindex(names).reset_index()
table['age'] = table['age'].apply(sorted)
num_samples = sample_data.groupby('age_group2').size().reindex(names).reset_index()

table_final = pd.concat([table, num_samples.iloc[:, 1]], axis=1)
table_final.columns = ['age group', 'age', 'num samples']

latex_table = table_final.to_latex(path_result+'brainspan_summary_rna_compact.tex',
                              index=False,
                              caption="", 
                              label="tab:brainspan_summary",)


################
# extended maps
extended_maps = ['complex1', 'complex2',
       'complex3', 'complex4','atpsynth', 
       'fa_metabolism', 'glycogen_metabolism', 
       'pdc', 'mas', 'gps', 'creatine', 'ros_detox', 'ros_gen',
       'no_signalling', 'atpase', 'gln_glu_cycle']


fig_size = {'age_group': (8, 6), 'age_group2': (8, 6)}
widths = {'age_group': 0.4, 'age_group2': 0.4}
rotations = {'age_group': 35, 'age_group2': 35}
categories = ['age_group', 'age_group2']

for category in categories:
    fig, axes = plt.subplots(4, 4, figsize=(fig_size[category]), sharex=True, sharey=False)
    axes = axes.flatten()
    
    for ax, key in zip(axes, extended_maps):
        sns.boxplot(data=df,
            x=category,
            y=key,
            boxprops={'facecolor': 'none', 'edgecolor': 'darkgrey'},
            whiskerprops={'color': 'darkgrey'},
            capprops={'visible': False},
            medianprops={'color': 'crimson'},
            showfliers=False,
            linewidth=0.7,
            width=widths[category],
            ax=ax,)
        
        sns.stripplot(data=df,
            x=category,
            y=key,
            color='sandybrown',
            size=3,
            alpha=0.4,
            jitter=True,
            zorder=-5,
            ax=ax,)
        
        sns.lineplot(data=df,
            x=category,
            y=key,
            estimator=np.median,
            ci=None,
            sort=False,
            color='mediumvioletred',
            marker='o',
            markersize=0.7,
            markeredgewidth=0,
            ax=ax,
            linewidth=0.7,)
        
        ax.set_title(key)
        ax.set_xlabel(None)
        ax.set_ylabel('expression (normalized RPKM)')
        sns.despine(ax=ax)
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotations[category], 
                ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.xticks(rotation=rotations[category], ha='right', rotation_mode='anchor')
    plt.savefig(path_fig + category + '_box_line_rpkm_extended_log2zscore.svg')
    plt.show()


# smoothed curves
# convert to long format
df_long = pd.melt(df, id_vars=['log_age_days'], value_vars=extended_maps, 
                  var_name='pathway', value_name='expression')

df_long['pathway'] = pd.Categorical(df_long['pathway'], 
                                    categories=extended_maps, ordered=True)

# make r dataframe
df_r = pandas2ri.py2rpy(df_long)

# plot
grdevices.svg(path_fig+"brainspan_geom_loess_extended.svg", width=8, height=6)

pp = (ggplot2.ggplot(df_r) +
      ggplot2.aes_string(x='log_age_days', y='expression') +
      ggplot2.geom_point(color='sandybrown', size=1.5, stroke=0,
                         alpha=0.4,) +
      ggplot2.geom_smooth(method="loess", se=False, 
                          color='mediumvioletred', size=0.6) +
      ggplot2.facet_wrap("~pathway", scales='free') +
      ggplot2.theme(axis_line=ggplot2.element_line(color='black'),
                    panel_background=ggplot2.element_blank(),
                    panel_grid_major=ggplot2.element_blank(),
                    panel_grid_minor=ggplot2.element_blank()) +
                    ggplot2.geom_vline(xintercept=np.log10(40*7), 
                                       color='grey', linetype='dashed', 
                                       size=0.7, show_legend=False))
pp.plot()
grdevices.dev_off()


#############################
# trajectory of dev processes

# read brainspan normalzied data
df_norm = pd.read_csv(path_result+'brianspan_exp_rna_uqnorm.csv').iloc[:, 1:]

final_df = df_norm.drop('donor_id', axis=1).T.reset_index(names='gene_symbol')

# gene sets
# kang genesets
# load cell type genes
kang_sets = pd.read_csv(path_data+'kang_genesets.csv')
kang_genes = kang_sets.groupby('Functional group')['Gene symbol'].apply(list).reset_index()
kang_genes_dict = dict(zip(kang_genes['Functional group'], kang_genes['Gene symbol']))

# load cell type genes
cells = pd.read_csv(path_data+'li2018_celltypes.csv')
li_genes = cells.groupby('Cell type')['Gene symbol'].apply(list).reset_index()
li_genes_dict = dict(zip(li_genes['Cell type'], li_genes['Gene symbol']))

geneset_dict = {'NPC': li_genes_dict['NPC_prenatal'],
                'synapse_dev': ['SYP', 'SYPL1', 'SYPL2', 'SYN1']}

# plot
# get cell type expression
bs_exp_cells = {}
bs_mean_cells = {}
for key, value in geneset_dict.items():
        bs_exp_cells[key] = final_df[final_df.gene_symbol.isin(value)]
        bs_mean_cells[key] = np.mean(bs_exp_cells[key].iloc[:, 1:], axis=0)

final_genes_rna = {}
for key, value in bs_exp_cells.items():
    final_genes_rna[key] = list(value.gene_symbol)

#{'NPC': ['ENO1', 'RPLP0', 'RPL3', 'RPL18A', 'GAPDH', 'TPI1', 'RPL5', 
# 'HNRNPA1', 'RPL7', 'RPL7A', 'EEF1A1', 'HMGB2', 'RPSA', 'NPM1', 'ACTG1', 
# 'EEF1A1P5', 'GNB2L1', 'RP4-604A21.1', 'TPI1P1', 'RPL41', 'HMGN2P5', 
# 'RPL13AP5']
# 'synapse_dev': ['SYN1', 'SYPL1', 'SYP']}

# converting mean expression dict to dataframe
bs_mean_cells_df = (pd.DataFrame.from_dict(bs_mean_cells)).dropna(axis=1)
# add sample info for plotting
df = pd.concat([sample_data.reset_index(drop=True), bs_mean_cells_df], axis=1)

# plot
fig_size = {'age': (6, 5), 'age_group2': (6, 5)}
widths = {'age': 0.4, 'age_group2': 0.4}
rotations = {'age': 90, 'age_group2': 35}
categories = ['age', 'age_group2']

for category in categories:
    fig, axes = plt.subplots(3, 3, figsize=(fig_size[category]), sharex=True, sharey=False)
    axes = axes.flatten()
    
    for ax, key in zip(axes, bs_mean_cells_df.columns):
        sns.boxplot(data=df,
            x=category,
            y=key,
            boxprops={'facecolor': 'none', 'edgecolor': 'darkgrey'},
            whiskerprops={'color': 'darkgrey'},
            capprops={'visible': False},
            medianprops={'color': 'crimson'},
            showfliers=False,
            linewidth=0.7,
            width=widths[category],
            ax=ax,)
        
        sns.stripplot(data=df,
            x=category,
            y=key,
            color='sandybrown',
            size=3,
            alpha=0.4,
            jitter=True,
            zorder=-5,
            ax=ax,)
        
        sns.lineplot(data=df,
            x=category,
            y=key,
            estimator=np.median,
            ci=None,
            sort=False,
            color='mediumvioletred',
            marker='o',
            markersize=0,
            markeredgewidth=0,
            ax=ax,
            linewidth=0.7,)
        
        ax.set_title(key)
        ax.set_xlabel(None)
        ax.set_ylabel('expression (normalized RPKM)')
        sns.despine(ax=ax)
        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotations[category], 
                        ha='right', rotation_mode='anchor')
            if ax != axes[0]:
                 ax.set_ylabel(None)
    plt.tight_layout()
    plt.savefig(path_fig + category + '_dev_trajectories_rna.svg')
    plt.show()