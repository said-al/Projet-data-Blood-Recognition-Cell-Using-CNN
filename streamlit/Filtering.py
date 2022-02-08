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
    if composant in ['H','S','V']:
        img = cv2.imread(img,cv2.IMREAD_COLOR)
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
        
    return th


# EMILIEN -- FILTERING FUNCTIONS

def filter_color_threshold(img):
    """
    Cette fonction à travers un mask, permet de filtrer une partie de l'image en sélectionnant par
    un seuil bas et un seuil haut, une plage de couleur. Toute plage de couleur en dehors du mask, 
    est remplacé par une couleur de fond (ici noir)
    L'image de sortie ne laisse apparaitre que la couleur entre les plages de couleurs
    """

    img = cv2.imread(img,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                                                    

    # définition des plages de couleurs dans l'espace colorimétrique que vous souhaitez afficher
    lower_color = np.array([30, 0, 0])
    upper_color = np.array([165,255,255])
    
    # Le mask ne laisse passer que des couleurs spécifiques entre les plages
    mask_color = cv2.inRange(img, lower_color, upper_color)
    
    # superposition de l'image d'origine et du mask.
    img_recovered = cv2.bitwise_and(img,img, mask= mask_color)

    return img_recovered
    

def filter_MeanShift(img):
    """
    Cette fonction qui utilise un MeanShift définit un nombre de clusters sur l'image 
    à l'aide de la fonction fournie "estimate_bandwidth". Ces clusters regroupent les pixel sur une plage RGB (centroid).
    La valeur des centroids est ensuite attribué à chaque pixel du cluster associé. 
    L'image reformé est donc constitué de N couleur, correspondant aux nombre de clusters.
    """

    img = cv2.imread(img,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # On applique un filtre pour réduire le possible bruit de l'image
    img = cv2.medianBlur(img, 3)

    # On aplatit l'image
    flat_image = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=0.5, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth,max_iter=2000,bin_seeding=True)
    ms = ms.fit(flat_image)
    labels=ms.labels_                                   # Pour chaque pixel, on récupere le label estimé.
    centroids = ms.cluster_centers_                     # On recupère les centroids ([R G B] moyen pour chaque)

    centroids = np.uint8(centroids)
    result = centroids[labels]
    result = result.reshape(img.shape)

    return result

def filter_Kmeans2(img,n_clusters=3):
    """
    img : image en BGR
    Description:Cette fonction est composée de deux étapes. La première est, à travers un mask, de
filtrer une partie de l'image en sélectionnant par un seuil bas et un seuil haut, une plage de couleur. Toute plage de couleur en dehors du mask est remplacé par une couleur de fond (ici, le noir).
La seconde étape est d’y appliquer un Kmeans avec k-centroids = 3, permettant de regrouper les couleurs de l'image en 3 groupes. 
L’objectif étant de retrouver sur l’image, les couleurs définies par les 3 centroïdes, la couleur du fond, la couleur de la cellule et la couleur de son noyau.
    """     
    
    # Filtrage du background par couleur
    img = cv2.imread(img,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([30, 0, 0])
    upper_color = np.array([165,255,255])
    mask_color = cv2.inRange(img, lower_color, upper_color)
    filtered_image = cv2.bitwise_and(img,img, mask= mask_color)

    # img = cv2.resize(filtered_image[index_filtered_img],(360,360))
    img = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    pixel_vals = img.reshape((-1,3)) 
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
    retval, labels, centers = cv2.kmeans(pixel_vals, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] 
    segmented_image = segmented_data.reshape(img.shape)
    
    return segmented_image


# SAID -- FILTERING FUNCTIONS

def EqualizerImg(img) :
    '''
    Cette fonction permet de lire l'image en couleur ensuite la transformer dans différents codages couleurs. Principalement
    YUV, Y représente la luminance (informations de luminosité) tandis que les deux autres (U et V) sont des données de 
    chrominance (informations de couleur). ensuite on égalise ((amélioration du contraste)) l'image en YUV. 
    
    après ces étapes on appliquera un filtre pour débruiter l'image qui est le filtre non-local means.
    --------------------------------------------------------------------
    paramètre :
        img : c'est le path où aller recupérer l'image + le nom de l'image en question

    '''
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    # Egalisation de l'image
    img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_equ = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Appliquer le filtre 'non-local means filter' sur img
    dst_img = cv2.fastNlMeansDenoisingColored(src=img_equ,
                                              dst=None,
                                              h=10,
                                              hColor=10,
                                              templateWindowSize=7,
                                              searchWindowSize=21)
    return dst_img

def RemoveBackground(img):
    """
    Cette fonction permis de nettoyer le background d'une image originale, elle procède comme suite au départ on éffectue un 
    GaussianBlur et en esnuite un threshold avec les résultat de threshold on crée un mask avec morphologyEx, qu'on appliquera 
    à notre image. 
    
    paramètre :: 
    img : notre image original en GBR 
    
    """
    #start = time.time()
    
    l = int(max(5,6))
    u = int(max(6,6))
    
    edges = cv2.GaussianBlur(img, (21, 51), 3) # (21, 51), 3
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(edges, l, u)
    
    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #(5, 5)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    data = mask.tolist()
    sys.setrecursionlimit(10**9) #(10**8)
    for i in  range(len(data)):
        for j in  range(len(data[i])):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
        for j in  range(len(data[i])-1, -1, -1):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
    image = np.array(data)
    image[image !=  -1] =  255
    image[image ==  -1] =  0

    mask = np.array(image, np.uint8)

    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask ==  0] =  255
    
    
    #end = time.time()
    #elapsed_time = end - start
    
    return result #, elapsed_time

# Fonction pour récupérer k images de manière aléatoire
def RandomImage(k,noms_classes):
    '''
    la fonction RandomImage va nous permettre de récupérer un nombre d'images (k) de chaque classes de nos cellules
    de manière aléatoire, ensuite les stocker dans une liste img_random 
    
    paramètres :
    noms_classes : les noms de nos 8 calsses de cellules sanguines
    k : le nombre d'images qu'en veut récupérer de chaques classes de notre data base
    '''
    img_random = []
    for i in range(len(noms_classes)) :
        for j in range(k):
            img = random.choice(noms_classes[i])
            img_random.append(img) 
    return img_random


# PAUL -- FILTERING FUNCTIONS

def filter_Kmeans1(img,n_clusters=6,bckgrd=False):
    """
    img : image in BGR
    
    Description:
    Cette fonction applique un Kmeans pour K=6 qui regroupe les couleurs de l'image en 6 groupes, puis, parmi les centroids calculés, 
    on redéfinit 2 groupes de centroïdes en forçant le départ de l'algorithme à une couleur proche du fond et de la couleur des globules rouges.
    Parmi les 2 groupes de couleurs obtenus, le premier groupe correspondra donc au fond et aux globules rouges, que l'on fait apparaître en blanc.

    """
    
    # Convert image in RGB
    img = cv2.imread(img,cv2.IMREAD_COLOR)
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

    centroids = np.uint8(centroids)

    X_recovered=centroids[labels]                                                             # set-up each pixel to be equal to the centroid of its cluster
    img_recovered = X_recovered.reshape(h,w,c)                                                # reshape the array to the image initial dimensions                                                       # elapsed time

    return img_recovered


def filter_KmeansXYRGB(img,n_clusters=10):
    """
    img : image in BGR
    
    Description:
    Cette fonction applique un Kmeans pour K=n_clusters qui regroupe les pixels par positions sur l'image et les 3 canaux de couleurs RGB
    A chaque pixel est appliqué ses coordonnées X, Y ainsi que ses intenistés lumineuses R,G,B.
    On soumet donc au Kmeans une matrice (363, 360, 5) où 5 représente le nombre de dimensions pour X,Y,R,G,B.
    
    """
    
    # start = time.time()                                                        # start timer
    
    # Convert image in RGB
    img = cv2.imread(img,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h,w,c=img.shape                                                            # get image dimensions
    XY = np.zeros([h,w,2])
    for x in range(h):                                                         # create matrix h,w,2 which gives X and Y coordinates of the pixel
        for y in range(w):
            XY[x,y] = [x,y]
    
    XY_img = np.concatenate([XY,img],axis=2)                                   # Concatenate the image with coordinates to create a matrix h,w,c+2 (c+2 = 5 coordinates X,Y,R,G,B)
    X = XY_img.reshape(h*w,c+2)                                                # reshape the image to an array (nb_pixels x canals)
    cluster=KMeans(n_clusters=n_clusters)                                      # KMeans instanciation
    cluster.fit(X)                                                             # KMeans training
    labels=cluster.labels_                                                     # Label for each pixel
    centroids = cluster.cluster_centers_                                       # K centroids

    centroids = centroids.astype(int)
    X_recovered=centroids[labels]                                              # set-up each pixel to be equal to the centroid of its cluster
    XY_img_recovered = X_recovered.reshape(h,w,c+2)                            # reshape the array to the XY_image dimensions
    img_recovered = XY_img_recovered[:,:,-3:]                                  # we get only RGB colors to reconstitute images
    
    img_recovered = img_recovered.astype('uint8')
    
    # end = time.time()                                                          # stop timer
    # elapsed_time = end - start                                                 # elapsed time
    
    return img_recovered