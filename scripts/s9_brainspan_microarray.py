

'''
BrainSpan microarray analysis
1. cleanup
2. plot trajectories

Author: Moohebat
Data: 13/12/2024

'''
# import packages
import pickle
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
path_bs = path_data+'brainspan/gene_array_matrix_csv/'

# data was downloaded from brainspan "Exon microarray summarized to genes"
# expression matrix
bs_exp = pd.read_csv(path_bs + 'expression_matrix.csv', header=None) # (gene x sample)
bs_exp = bs_exp.iloc[:, 1:] # (17604 x 492)

# gene metadata
gene_data = pd.read_csv(path_bs + 'rows_metadata.csv')['gene_symbol']

# sample metadata
sample_data = pd.read_csv(path_bs + 'columns_metadata.csv').set_index('column_num')

#simple stats
np.unique(sample_data['donor_id']).shape #35 donors
np.unique(sample_data['age']).shape #27 developmental stages
np.unique(sample_data['structure_acronym']).shape #26 structure acronyms

order = ['8 pcw', '9 pcw', '12 pcw', '13 pcw', 
         '16 pcw', '17 pcw',  '19 pcw', '21 pcw', 
         '24 pcw', '25 pcw', '26 pcw','4 mos', '10 mos', 
         '1 yrs', '2 yrs', '3 yrs', '4 yrs', '8 yrs',
         '13 yrs', '15 yrs','18 yrs','21 yrs', '23 yrs', 
         '30 yrs', '36 yrs', '37 yrs', '40 yrs', ]

# adding age category
def map_to_category(list_names, list_items):
    map_to_category = {item: category for category, items in \
                       zip(list_names, list_items) for item in items}
    return map_to_category

# broad age category Li et al 2018
fetal = ['8 pcw', '9 pcw', '12 pcw', '13 pcw', '16 pcw', 
         '17 pcw', '19 pcw', '21 pcw','24 pcw', '25 pcw', 
         '26 pcw', '35 pcw', '37 pcw']
infant = ['4 mos','10 mos','1 yrs']
child = ['2 yrs', '3 yrs','4 yrs','8 yrs']
adolescent = ['11 yrs', '13 yrs', '15 yrs', '18 yrs', '19 yrs']
adult = ['21 yrs', '23 yrs','30 yrs', '36 yrs', '37 yrs',
        '40 yrs']

groups = [fetal,infant,child,adolescent,adult]
names = ['fetal','infant','child','adolescent','adult']
# map
sample_data['age_group'] = sample_data['age'].map(map_to_category(names, groups))

# broad age category kang et al 2011
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

# age periods Kang et al 2011
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

periods = [p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12, p13, p14]
names = ['p2','p3','p4','p5','p6','p7','p8','p9',
         'p10','p11','p12', 'p13', 'p14']
# map
sample_data['period'] = sample_data['age'].map(map_to_category(names, periods))

# adding cortex, non_cortex division
regions = np.unique(sample_data['structure_name']) # 26
# cortex and subcoretx divisions
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

names = ['cortex', 'subcortex']
ctx = [cortex, subcortex]
# map
sample_data['ctx'] = sample_data['structure_name'].map(map_to_category(names, ctx))

# anatomical divisions
front = ['MFC', 'DFC', 'OFC', 'VFC', 'M1C']
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


###########
# QC
# sample data cleanup
# region qc
# focusing on cortical regions
sample_data2 = sample_data[sample_data['ctx'] == 'cortex'] #345
# sample_data2 = sample_data
# 1. keep regions with at least 1 sample in each age group
np.unique(sample_data2['structure_acronym']).shape  # 26 unique regions
groups = sample_data2.groupby(['structure_acronym', 
                              'age_group']).size().unstack(fill_value=0)

structures = groups[(groups >= 1).all(axis=1)].index
sample_data2 = sample_data2[sample_data2['structure_acronym'].isin(structures)]
dropped = list(set(sample_data['structure_acronym']) - set(sample_data2['structure_acronym']))
# dropped ['M1C-S1C', 'CGE', 'MD', 'Ocx', 'CBC', 'AMY', 
# 'TCx', 'DTH', 'LGE', 'HIP', 'PCx', 'STR', 'MGE', 'URL', 'CB']
# 334 samples left
# if i run this on the whole brain: 465 samples left

# drop regions with low sample size
# np.unique(sample_data2['age']).shape  # 27
# sample_data3 = sample_data2.groupby('age').filter(lambda x: len(x) >= 6)
# np.unique(sample_data3['age']).shape  # 24
# dropped_ages = list(set(sample_data2['age']) - set(sample_data3['age']))
# # dropped ['9 pcw', '25 pcw', '26 pcw']

bs_exp = bs_exp.loc[:, sample_data2.index]
# 17604 x 334

##############
# gene cleanup
# 1. drop duplicate genes
gene_uniq = gene_data.drop_duplicates()
bs_exp = bs_exp.iloc[gene_uniq.index, :] # (17282 rows x 334 columns)

# 2. drop non-expressed genes
# excluded genes with a log2-transformed expression value <6
# this keeps 3673 genes
bs_exp = bs_exp.loc[(bs_exp >= 6).any(axis=1)] # 13635
# if whole brain 14139 genes by 465 samples
gene_qc = gene_uniq.loc[bs_exp.index]

bs_df = pd.concat([gene_qc, bs_exp], axis=1)

# save the final expression matrix
sample_data2.to_csv(path_result+'sample_data_micro_qc.csv')
bs_df.to_csv(path_result+'brainspan_micro_exp_qc.csv')

# quartile normalize

# load brainspan data
sample_data = pd.read_csv(path_result+'sample_data_micro_qc.csv').set_index('column_num')
bs_df = pd.read_csv(path_result+'brainspan_micro_exp_qc.csv').iloc[:, 1:]

df = pd.concat([sample_data['donor_id'].reset_index(drop=True), 
                bs_df.set_index('gene_symbol').T.reset_index(drop=True)], 
                axis=1)

percentile_75 = df.groupby('donor_id').apply(lambda group: np.percentile(group.iloc[:, 1:].values.flatten(), 75))
mean_p75 = np.mean(percentile_75)

df_norm = pd.DataFrame()
for donor in np.unique(sample_data['donor_id']):
    group = df.groupby('donor_id').get_group(donor)
    percent75 = np.percentile(group.iloc[:, 1:].values.flatten(), 75)
    group.iloc[:, 1:] = (group.iloc[:, 1:] / percent75) * mean_p75
    df_norm = pd.concat([df_norm, group], axis=0)

df_norm = df_norm.sort_index()

# save
df_norm.to_csv(path_result+'brianspan_exp_micro_uqnorm.csv')

#################
# energy analysis
# read brainspan normalzied data
sample_data = pd.read_csv(path_result+'sample_data_micro_qc.csv').set_index('column_num')
df_norm = pd.read_csv(path_result+'brianspan_exp_micro_uqnorm.csv').iloc[:, 1:]
final_df = df_norm.drop('donor_id', axis=1).T.reset_index(names='gene_symbol')

# loading energy pathway data
# for seperate curation based on brainspan data
with open(path_result+'energy_genelist_dict.pickle', 'rb') as f:
    energy_genes = pickle.load(f)

# fix ATP synthase gene symbols
atp_genes = ['ATP5A1', 'ATP5B', 'ATP5C1', 'ATP5D',
             'ATP5E', 'ATP5I', 'ATP5J2', 'ATP5L', 
             'ATP5F1', 'ATP5H', 'ATP5J', 'ATP5O']
energy_genes['atpsynth'] = np.array(atp_genes)

# last 14 genes in oxphos list are atpsynth genes
oxphos_genes = energy_genes['oxphos'][:-14]
oxphos_genes = np.append(oxphos_genes, atp_genes)
energy_genes['oxphos'] = oxphos_genes

# get brainspan expression matrix and mean for energy genes
bs_exp_energy = {}
bs_mean_energy = {}
for key, value in energy_genes.items():
        bs_exp_energy[key] = final_df[final_df.gene_symbol.isin(value)]
        bs_mean_energy[key] = np.mean(bs_exp_energy[key].iloc[:, 1:], axis=0)

# converting mean expression dict to dataframe
bs_mean_energy_df = pd.DataFrame.from_dict(bs_mean_energy)
df = pd.concat([sample_data.reset_index(drop=True), bs_mean_energy_df], axis=1)

#focusing on main energy pathways
energy_main =['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate', 'kb_util']

# plot
# box + strip + line
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
            markersize=3,
            markeredgewidth=0,
            ax=ax,
            linewidth=0.6,)
        
        ax.set_title(key)
        ax.set_xlabel(None)
        ax.set_ylabel('expression (signal intensity)')
        sns.despine(ax=ax)
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotations[category], 
                 ha='right', rotation_mode='anchor')
    plt.xticks(rotation=rotations[category], ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.savefig(path_fig + category + '_box_line_micro_zscore.svg')
    plt.show()


#################
# smoothed curves

# add log(age) column
df['log_age_days'] = np.log10(df['age_days'].values.astype('float'))

# convert to long format
df_long = pd.melt(df, id_vars=['log_age_days'], value_vars=energy_main, 
                  var_name='pathway', value_name='expression')

df_long['pathway'] = pd.Categorical(df_long['pathway'], 
                                    categories=energy_main, ordered=True)

# make r dataframe
df_r = pandas2ri.py2rpy(df_long)

# plot
# make grid
grdevices.svg(path_fig+"brainspan_geom_loess_micro.svg", width=8, height=5)

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


# strip for all ages
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


# simple stats
# plotting basic stats for brainspan data
# after qc
variables = ['donor_name', 'age', 'structure_acronym', 'gender']
for var in variables:
    plt.figure(figsize=(4,4))
    plot = sns.histplot(sample_data2, x=sample_data2[var], edgecolor='w', color='sandybrown')
    for p in plot.patches:
        plot.text(
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
            '%d' % int(p.get_height()),
            fontsize=5,
            color='k',
            ha='center',
            va='bottom',
        )
    plt.xlabel(var, fontsize=5)
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks([])
    plt.ylabel('number of samples', fontsize=5)
    plt.title('number of samples for each '+var, fontsize=5)
    sns.despine()
    plt.tight_layout()
    # plt.savefig(path_fig + var+'_sample_stat_qc.svg')
    plt.show()


# save summary of brainspan into latex table
num_samples = sample_data.groupby('donor_name').size()

names = ['early_fetal','mid_fetal','late_fetal','infancy', 
         'early_child','late_child','adolescent','adult']

table = sample_data.groupby('age_group2')['age'].apply(set).reindex(names).reset_index()
table['age'] = table['age'].apply(sorted)
num_samples = sample_data.groupby('age_group2').size().reindex(names).reset_index()

table_final = pd.concat([table, num_samples.iloc[:, 1]], axis=1)
table_final.columns = ['age group', 'age', 'num samples']

latex_table = table_final.to_latex(path_result+'brainspan_summary_micro_compact.tex',
                              index=False,
                              caption="", 
                              label="tab:brainspan_summary",)