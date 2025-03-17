
'''
gene set curation
Author: Moohebat
Date: 22/11/2024
'''
#getting gene sets using biomart. both for go and reactome.
#installing Go.db for getting go child terms
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("GO.db")

#loading libraries
library(dplyr)
library(biomaRt)
library(GO.db)
library(openxlsx)

#biomart version
biomart_v <- packageVersion("biomaRt") #2.50.3

#ensembl version
available_datasets <- listMarts() # 113

#loading ensembl human gene annotation data
hsapiens_ensembl_genes <- useMart("ensembl",
                                  dataset="hsapiens_gene_ensembl")

#list attributes cause i need to retrieve and filter by this
ensembl_attr <- listAttributes(hsapiens_ensembl_genes)
ensembl_filt <- listFilters(hsapiens_ensembl_genes)

#####################################
#getting gene sets from Gene Ontology
go_pathway <- c('glycolysis', 'ppp', 
                'tca', 'complex1', 
                'complex2', 'complex3',
                'complex4', 'etc', 
                'atpsynth', 'lactate_metabolism', 
                'lactate_transport', 'kb_util', 
                'betaox', 'glycogen_cat', 
                'glycogen_synth', 'fa_metabolism',
                'kb_metabolism', 'glycogen_metabolism',
                'ros_response', 'acetylcoa',
                'kb_synth', 'fa_synth',
                'superoxide_gen', 'superoxide_remove',
                'mas', 'gps', 
                'ros_gen')

go_pathway_id <- c('GO:0006096', 'GO:0006098', 
                   'GO:0006099', 'GO:0006120', 
                   'GO:0006121', 'GO:0006122',
                   'GO:0006123', 'GO:0019646',
                   'GO:0046933', 'GO:0006089', 
                   'GO:0015727', 'GO:0046952', 
                   'GO:0006635', 'GO:0005980', 
                   'GO:0005978', 'GO:0006631',
                   'GO:1902224', 'GO:0005977',
                   'GO:0034614', 'GO:0006085',
                   'GO:0046951', 'GO:0006633',
                   'GO:0042554', 'GO:0019430',
                   'GO:0043490', 'GO:0006127',
                   'GO:1903409')

#function for retrieving gene set based on go_id
get_gene_sets_go <- function(go_ids, pathway) {
  
  energy_df <- list()
  
  #loop over go_ids 
  for (i in seq_along(go_ids)) {
    
    # get child terms
    child_terms <- as.character(GOBPOFFSPRING[[go_ids[i]]])
    
    # concat parent and child names
    terms <- c(go_ids[i], child_terms)
    
    # get gene name and description for each term
    data <- getBM(attributes = c('external_gene_name',
                                 'description', 'go_linkage_type'), 
                  filters = 'go', 
                  values = terms, 
                  mart = hsapiens_ensembl_genes)
    
    #keep unique genes
    uniq_genes <- dplyr:: distinct(data, external_gene_name, .keep_all = TRUE)
    
    df <- data.frame(uniq_genes)
    
    # add to dataframe
    energy_df[[pathway[i]]] <- df
  }
  
  return(energy_df)
}

energy_df_go <- get_gene_sets_go(go_pathway_id, go_pathway)

###############################
#Reactome curations
reactome_pathway = c('glycolysis', 'ppp', 
                     'tca', 'etc', 
                     'atpsynth', 'kb_util',
                     'betaox', 'glycogen_cat', 
                     'glycogen_synth', 'ros_detox',
                     'fa_metabolism', 'no_signalling',
                     'kb_metabolism', 'glycogen_metabolism',
                     'kb_synth', 'fa_synth',
                     'mas', 'gps')

reactome_pathway_id = c('R-HSA-70171', 'R-HSA-71336', 
                        'R-HSA-71403', 'R-HSA-611105', 
                        'R-HSA-163210', 'R-HSA-77108', 
                        'R-HSA-77289', 'R-HSA-70221', 
                        'R-HSA-3322077', 'R-HSA-3299685',
                        'R-HSA-8978868', 'R-HSA-392154',
                        'R-HSA-74182', 'R-HSA-8982491',
                        'R-HSA-77111', 'R-HSA-75105',
                        'R-HSA-9856872', 'R-HSA-188467')


#function for retrieving gene set based on reactome
get_gene_sets_reactome <- function(reactome_ids, pathway) {

  energy_df <- list()
  
  #loop over go_ids 
  for (i in seq_along(reactome_ids)) {
    #retrieve gene name and description
    genes <- getBM(attributes=c('external_gene_name',
                                'description'),
                   filters = 'reactome',
                   values = reactome_ids[i],
                   mart = hsapiens_ensembl_genes)
    
    df <- data.frame(genes)
    
    # Store the data frame in the list
    energy_df[[pathway[i]]] <- df
  }
  
  # add to dataframe
  return(energy_df)
}

energy_df_reactome <- get_gene_sets_reactome(reactome_pathway_id, 
                                             reactome_pathway)

###############################
#getting shared genes between reactome and GO gene sets

get_intersect<- function(pathways) {
  
  shared_genes_dfs <- list()
  
  for (pathway in pathways) {
    
    # keep shared genes
    shared_genes <- intersect(energy_df_go[[pathway]]$external_gene_name, 
                              energy_df_reactome[[pathway]]$external_gene_name)
    
    df_shared <- energy_df_go[[pathway]][energy_df_go[[pathway]]$external_gene_name %in% shared_genes, ]

    shared_genes_dfs[[pathway]] <- df_shared
  }

  return(shared_genes_dfs)
}

# run
shared_pathways = c('glycolysis', 'ppp', 
             'tca', 'etc', 
             'atpsynth', 'kb_util', 
             'betaox', 'glycogen_cat', 
             'glycogen_synth', 'fa_metabolism',
             'kb_metabolism', 'glycogen_metabolism',
             'kb_synth', 'fa_synth',
             'mas','gps')

shared_df <- get_intersect(shared_pathways)

#saving gene sets
go_path = "./results/energy_sets_new/go/"
for (pathway in names(energy_df_go)) {
  write.csv(energy_df_go[[pathway]], 
            file = paste0(go_path, pathway, ".csv"), 
            row.names = FALSE)
}

reactome_path = "./results/energy_sets_new/reactome/"
for (pathway in names(energy_df_reactome)) {
  write.csv(energy_df_reactome[[pathway]], 
            file = paste0(reactome_path, pathway, ".csv"), 
            row.names = FALSE)
}

shared_path = "./results/energy_sets_new/shared/"
for (pathway in names(shared_df)) {
  write.csv(shared_df[[pathway]], 
            file = paste0(shared_path, pathway, ".csv"), 
            row.names = FALSE)
}


########################
#enzyme complexes
#Na/K ATPase, GO:0005890
#pyruvate dehydrogenase complex, GO:0045254
#pyruvate dehydrogenase kinase, GO:0004740
#pyruvate dehydrogenase phosphatase, GO:0004741
#pyruvate carboxylase, GO:0004736
#creatine_kinase, GO:0004111

enzymes <- c('atpase', 'pdc', 'pdk', 'pdp', 'pc', 'ck')
enzymes_id <- c('GO:0005890', 'GO:0045254', 'GO:0004740',
                'GO:0004741', 'GO:0004736', 'GO:0004111')

enzyme_df_go <- get_gene_sets_go(enzymes_id, enzymes)

go_path = "./results/energy_sets_new/go/"
for (enzyme in names(enzyme_df_go)) {
  write.csv(enzyme_df_go[[enzyme]], 
            file = paste0(go_path, enzyme, ".csv"), 
            row.names = FALSE)
}
