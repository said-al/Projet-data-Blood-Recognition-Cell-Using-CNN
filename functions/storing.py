import warnings
import time
import pandas as pd
import numpy as np
import cv2 # import OpenCV
import matplotlib.pyplot as plt
from tqdm import tqdm #pour barre de temps dans une boucle for
from functions.Filtering import OTSU_threshold, filter_color_threshold,filter_Kmeans1,filter_Kmeans2,filter_MeanShift

def database_generate(df, sample_per_cell_type, resize_dim, filter_option, savename):
    """
    Description : this function saves a csv file after applying all preprocessing parametered
    """
    
    # pour ignorer les warnings dans le cas du Mean Shift
    if filter_option == 'MeanShift':
        warnings.filterwarnings("ignore")

    npc = sample_per_cell_type # nombre d'images par catégorie de cellule à scanner et filtrer
    #filter options : 'Kmeans1','Kmeans2','OTSU_S','OTSU_L','MeanShift'

    # On récupère les dimensions du format d'image le plus fréquent:
    # h=int(df.img_dim.mode()[0].split(' x ')[0])
    # w=int(df.img_dim.mode()[0].split(' x ')[1])
    (h,w)=resize_dim

    # Création d'une liste d'index à scanner dans le dataframe
    cell_types = df.cell_type_code.value_counts().index # cell types sorted by number of apparitions in the dataset

    # index selection:
    selected_index=[]
    for ct in cell_types: 
        all_index_ct=df[df.cell_type_code==ct].index
        new_selected_index=np.random.choice(all_index_ct, size=npc) # on en choisit npc par cell_type (npc= nombre de cellules)
        selected_index=np.concatenate([selected_index,new_selected_index]).astype(int)
        

    # On process pour chacun de ces indexes:

    data =[]

    start = time.time()

    # tqdm pour la barre d'avancement
    for i in tqdm(selected_index):

        p_start = time.time()
        
        filename =df.loc[i,'filename']
        filename = filename.replace('dataset\\','C:\\Users\\luap_\\OneDrive\\Data_Science\\Projects\\bloody_spy_blast\\dataset\\') # rediriger vers le dossier en local
        img_height =df.loc[i,'img_height'] # hauteur de l'image
        img_width =df.loc[i,'img_width']
        
        # img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(filename,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # on redimensionne l'image si elle n'est pas à la taille classique calculée ci-dessus
        if (img_height,img_width) != (h, w):
            # certaines images ne peuvent pas être redimensionnées d'où l'utilisation de try:
            try:
                # img = cv2.resize(img, dsize = (h,w), interpolation = cv2.INTER_LINEAR)
                img = cv2.resize(img,(h,w))
            except Exception as e:
                print('impossible to reshape',filename,'for the following reason:\n',str(e))
                # on supprime ce fichier de notre base de données car nous ne pourrons pas l'intégrer à notre modèle:
                df=df[df.filename != filename]
                print('The image was deleted from database.')
        
        if filter_option == 'Kmeans1':
            
            # convert with Kmeans filter
            img1, _ = filter_Kmeans1(img,n_clusters=6,bckgrd=False)

            # convert in grayscale
            rgb_weights = [1/3, 1/3, 1/3]  # [0.2989, 0.5870, 0.1140] => to play with gray shades
            img_gray = np.dot(img1[...,:3], rgb_weights)

        elif filter_option == 'Kmeans2':
            
            # convert with Kmeans filter
            img1, _ = filter_Kmeans2(img,n_clusters=4)

            # convert in grayscale
            rgb_weights = [1/3, 1/3, 1/3]  # [0.2989, 0.5870, 0.1140] => to play with gray shades
            img_gray = np.dot(img1[...,:3], rgb_weights)
        
        elif filter_option == 'OTSU_S':
            
            img_gray, _ = OTSU_threshold(img,composant='S',blur=False)
            
        elif filter_option == 'OTSU_L':
            
            img_gray, _ = OTSU_threshold(img,composant='L',blur=False)
            
        elif filter_option == 'MeanShift':
    
            # convert with Kmeans filter
            img1, _ = filter_MeanShift(img)

            # convert in grayscale
            rgb_weights = [1/3, 1/3, 1/3]  # [0.2989, 0.5870, 0.1140] => to play with gray shades
            img_gray = np.dot(img1[...,:3], rgb_weights)

        # convert in an array
        h,w=img_gray.shape
        X=img_gray.reshape(1,h*w)[0]
        
        # process time
        # p_end = time.time()
        # p_elapsed = p_end - p_start
        
        # label:
        img_name = [df.loc[i,'img_name']]
        # process_time=[round(p_elapsed,6)]
        label = [df.loc[i,'cell_type3']]
        
        # store in data
        # data.append(np.concatenate((img_name,process_time,label,X),axis=0))
        data.append(np.concatenate((img_name,label,X),axis=0))
        
        # print information on advancement
        # print(f'file processed : {filename}. Process time : {round(p_elapsed,2):0} s.')
        

    end = time.time()
    elapsed = end - start
    print(f'Time elapsed to preprocess: {elapsed//60 :0}min {round(elapsed % 60,0):0}sec')

    # Create dataframe with all flat images
    # col_names= np.concatenate( (['img_name','process_time','label'],['p'+str(i) for i in range(1,h*w+1)]),axis=0)
    col_names= np.concatenate( (['img_name','label'],['p'+str(i) for i in range(1,h*w+1)]),axis=0)
    data_df=pd.DataFrame(data,columns = col_names)
    # data_df.label.value_counts()
    
    # max_rows_csv = 1048576 # maxiumum of rows that a csv can support
    max_columns_csv = 16384 # maxiumum of columns that a csv can support
    
    if h*w > max_columns_csv:
        # on transpose le dataframe avant enregistrement
        (data_df.T).to_csv(savename)
    else:
        # on ne transpose pas le dataframe avant enregistrement.
        data_df.to_csv(savename) # si on a déjà transposé

    end2 = time.time()
    elapsed2 = end2 - end
    print(f'Time elapsed to save csv file: {elapsed2//60 :0}min {round(elapsed2 % 60,0):0}sec')

    return data_df, elapsed, elapsed2



def database_load(savename,size):
    
    start = time.time()
    
    # pour ignorer le problème de mémoire souvent rencontré
    warnings.filterwarnings("ignore")
    
    (h,w)=size
    
    # max_rows_csv = 1048576 # maxiumum of rows that a csv can support
    max_columns_csv = 16384 # maxiumum of columns that a csv can support
    
    if h*w > max_columns_csv:
        # on transpose le dataframe à la lecture
        data_df=pd.read_csv(savename,index_col=0).T
    else:
        # on ne transpose pas le dataframe à la lecture
        data_df=pd.read_csv(savename,index_col=0)
        
    end = time.time()
    elapsed = end - start
    print(f'Time elapsed to load {savename}: {elapsed//60 :0}min {round(elapsed % 60,0):0}sec')
    
    return data_df, elapsed