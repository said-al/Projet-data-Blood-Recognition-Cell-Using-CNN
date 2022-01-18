import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import cv2 # OpenCV to read images
from sklearn.cluster import MeanShift, estimate_bandwidth


# CAROLINE -- FILTERING FUNCTIONS

def OTSU_threshold(img,composant='S',blur=False):
    """
    img : image in BGR
    composant : 'H','S','V' if you want to threshold one of HSV composant
                or 'L','A','B' if you want to threshold one of LAB composant
    blur : if you want to blur image with a 
    
    Description : Cette fonction permet de faire un seuillage d'OTSU sur la composante choisie

    """
    
    start = time.time()                                                                       # start timer
    
    if composant in ['H','S','V']:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H,S,V= cv2.split(img)
        if composant == 'H':
            ret, th = cv2.threshold(H,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif composant == 'S':
            ret, th = cv2.threshold(S,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif composant == 'V':
            ret, th = cv2.threshold(V,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    if composant in ['L','A','B']:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L,A,B= cv2.split(img)
        if composant == 'L':
            ret, th = cv2.threshold(L,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif composant == 'A':
            ret, th = cv2.threshold(A,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif composant == 'B':
            ret, th = cv2.threshold(B,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
    if blur == True:
        th = cv2.GaussianBlur(th, (3,3),0)
    
    end = time.time()                                                                         # stop timer
    elapsed_time = end - start                                                                # elapsed time

    return th, elapsed_time



# EMILIEN -- FILTERING FUNCTIONS

def filter_color_threshold(img):
    """
    Cette fonction à travers un mask, filtre une partie de l'image (ici le noyau) des plages de couleurs dans l'espace colorimétrique que l'on souhaite afficher 
    L'image de sortie ne laisse apparaitre que la couleur entre les plages de couleurs et la couleurs filtrée est affiché en noir.
    """

    # start timer
    start = time.time()                                                                       

    # définition des plages de couleurs dans l'espace colorimétrique que vous souhaitez afficher
    lower_color = np.array([30, 0, 0])
    upper_color = np.array([165,255,255])
    
    # Le mask ne laisse passer que des couleurs spécifiques entre les plages
    mask_color = cv2.inRange(img, lower_color, upper_color)
    
    # superposition de l'image d'origine et du mask.
    img_recovered = cv2.bitwise_and(img,img, mask= mask_color)
    
    # stop timer
    end = time.time()         
    
    # elapsed time
    elapsed_time = end - start

    return img_recovered, elapsed_time
    

def filter_MeanShift(img):
    """
    Description:
    Mean Shift sur l'image
    """

    # start timer
    start = time.time()      
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([30, 0, 0])
    upper_color = np.array([165,255,255])
    mask_color = cv2.inRange(img, lower_color, upper_color)
    img = cv2.bitwise_and(img,img, mask= mask_color)

    # img = cv2.resize(img,(363,360))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    
    # filtre de réduction du bruit
    img = cv2.medianBlur(img, 3)

    # aplatir l'image
    flat_image = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=0.5, n_samples=100)
    ms = MeanShift(bandwidth, max_iter=2000, bin_seeding=True)
    ms.fit(flat_image)
    labels=ms.labels_
    centroids = ms.cluster_centers_
    
#     COMPRENDRE CETTE PARTIE

#     # Obtenir la couleur moyenne de chaque centroid
#     total = np.zeros((centroids.shape[0], 3), dtype=float)
#     count = np.zeros(total.shape, dtype=float)

#     # Pour chaque label dans le tableau des labels
#     for j, label in enumerate(labels):
#         #On ajoute les valeurs RGB de chaque pixel de l'image applatit
#         total[label] = total[label] + flat_image[j]
#         count[label] += 1

#     # On calcule la moyenne du RGB
#     avg = total/count
#     # On arrondi à l'entier
#     avg = np.uint8(avg)

#     # transposition de l'image étiquetée dans la couleur moyenne correspondante
#     result = avg[labels]
#     img_filtered = result.reshape((img.shape))

    
    result = centroids[labels].astype(int)
    img_filtered = result.reshape((img.shape))
    
    # stop timer
    end = time.time()         
    
    # elapsed time
    elapsed_time = end - start

    return img_filtered, elapsed_time


def filter_Kmeans2(img,n_clusters=3):
    """
    img : image en BGR
    Description:
    CV2 Kmeans
    """

    # start timer
    start = time.time()      
    
    # Filtrage du background par couleur
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([30, 0, 0])
    upper_color = np.array([165,255,255])
    mask_color = cv2.inRange(img, lower_color, upper_color)
    filtered_image = cv2.bitwise_and(img,img, mask= mask_color)

    #On recupere l'image (par son index) de la liste crée par filtrage par couleur et on l'affiche
    # img = cv2.resize(filtered_image[index_filtered_img],(360,360))
    img = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    pixel_vals = img.reshape((-1,3)) 
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
    retval, labels, centers = cv2.kmeans(pixel_vals, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] 
    segmented_image = segmented_data.reshape((img.shape)) 
    
    # stop timer
    end = time.time()         
    
    # elapsed time
    elapsed_time = end - start

    return segmented_image, elapsed_time

# SAID -- FILTERING FUNCTIONS





# PAUL -- FILTERING FUNCTIONS

def filter_Kmeans1(img,n_clusters=6,bckgrd=False):
    """
    img : image in BGR
    
    Description:
    Cette fonction applique un Kmeans pour K=6 qui regroupe les couleurs de l'image en 6 groupes, puis, parmi les centroids calculés, 
    on redéfinit 2 groupes de centroïdes en forçant le départ de l'algorithme à une couleur proche du fond et de la couleur des globules rouges.
    Parmi les 2 groupes de couleurs obtenus, le premier groupe correspondra donc au fond et aux globules rouges, que l'on fait apparaître en blanc.

    """
    
    start = time.time()                                                                       # start timer
    
    # Convert image in RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h,w,c=img.shape                                                                           # get image dimensions
    X=img.reshape(h*w,c)                                                                      # reshape the image to an array (nb_pixels x canals)
    cluster=KMeans(n_clusters=n_clusters)                                                     # KMeans instanciation
    cluster.fit(X)                                                                            # KMeans training
    labels=cluster.labels_                                                                    # Label for each pixel
    centroids = cluster.cluster_centers_                                                      # K centroids
    
    if bckgrd==False:
        centroids_init = np.array([[252,227,199]                                              # we force the algorithm to start with the label 0 centroid close to background color
                                   ,[69,24,130]])
        cluster2=KMeans(n_clusters=2,init=centroids_init,n_init=1).fit(centroids)             # divide centroids in 2 clusters
        centroids[cluster2.labels_==0] = [255,255,255]                                        # set up all centroids with label 0 to blank color
        # # centroids[cluster2.labels_==0] = cluster2.cluster_centers_[0]                     # set up all centroids with label 0 to their centroid

    centroids = centroids.astype(int)
    X_recovered=centroids[labels]                                                             # set-up each pixel to be equal to the centroid of its cluster
    img_recovered = X_recovered.reshape(h,w,c)                                                # reshape the array to the image initial dimensions
    
    end = time.time()                                                                         # stop timer
    elapsed_time = end - start                                                                # elapsed time

    return img_recovered, elapsed_time


