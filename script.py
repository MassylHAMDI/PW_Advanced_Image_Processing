import numpy as np
import skimage as ski
from scipy import ndimage
from skimage.feature import peak_local_max
from scipy.signal import convolve2d

class Image_processing:
    def __init__(self):
        pass

    def convert_to_grayscale(self, input_img):
        """Converts the image to grayscale.

        Args:
            input_img (ndarray): L'image d'entrée à mettre en gris.

        Returns:
            ndarray: L'image en gris.
        """
        return ski.color.rgb2gray(input_img)
    

    def get_gradient(self,input_img):
        """_summary_

        Args:
            input_img (_type_): _description_

        Returns:
            _type_: _description_
        """
        Ix = np.gradient(input_img, axis = 1)
        Iy = np.gradient(input_img, axis = 0)
        return Ix,Iy
    
    
    def rotate_image(self, input_img, angle):
        """
            Effectue une rotation de l'image donnée selon l'angle spécifié.

            Args:
                input_img (ndarray): L'image d'entrée à faire pivoter.
                angle (float): L'angle de rotation en degrés.

            Returns:
                ndarray: L'image pivotée.
        """

        return ski.transform.rotate(input_img, angle)
    
    def remove_local_non_maxima(self, input_img, pnt_crit, size = 3):
        """
            Supprime les points d'intérêt locaux non maximaux dans une image.

            Args:
                input_img (ndarray): Image d'entrée sous forme de tableau NumPy.
                pnt_crit (ndarray): Tableau NumPy contenant les coordonnées des points d'intérêt à évaluer.
                size (int): Taille de la fenêtre à utiliser pour supprimer les points d'intérêt locaux non maximaux. Par défaut 3.

            Returns:
                ndarray: Tableau NumPy contenant les coordonnées des points d'intérêt locaux maximaux.

        """

        # Créer une nouvelle image
        new_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)

        # Créer un masque de booléens pour les points critiques
        mask = np.zeros(new_img.shape, dtype=bool)
        mask[tuple(zip(*pnt_crit))] = True

        # Placer des valeurs de 1 dans la nouvelle image aux emplacements des points critiques
        new_img[mask] = 1

         # Trouver les coordonnées des points d'intérêt locaux maximaux en utilisant la fonction peak_local_max de scikit-image
        local_non_maxima = peak_local_max(new_img, min_distance=size)
    
        return local_non_maxima
    

class Coins_detector(Image_processing):
    def __init__(self):
        super().__init__()
        pass

    def haris_detector(self, input_img, K, methods_wds, remover = False, size_wds = None, std_dev= None, suppression_size = 3):
        """
            Détecte les points d'intérêt dans une image en utilisant le détecteur de Harris.

            Args:
                input_img (ndarray): Image d'entrée.
                K (float): Paramètre de Harris qui contrôle la sensibilité à la différence de structure.
                methods_wds (str): Méthode de pondération de la fenêtre. Doit être "rectangulaire" ou "gaussienne".
                remover (bool): Si True, supprime les points d'intérêt locaux non maximaux. Par défaut False.
                size_wds (int): Taille de la fenêtre à utiliser pour la méthode "rectangulaire". Ignoré si la méthode est "gaussienne".
                std_dev (float): Écart type de la gaussienne à utiliser pour la méthode "gaussienne". Ignoré si la méthode est "rectangulaire".
                suppression_size (int): Taille de la fenêtre à utiliser pour supprimer les points d'intérêt locaux non maximaux. Ignoré si `remover` est False.

            Returns:
                ndarray: Carte des coins de Harris pour l'image d'entrée.
                ndarray: Tableau NumPy contenant les coordonnées des points d'intérêt détectés.

        """

        gray_img = self.convert_to_grayscale(input_img) # Convertir l'image en niveaux de gris

        Ix, Iy = self.get_gradient(gray_img)   # Calculer les gradients horizontaux et verticaux de l'image

        A, B, C = Ix**2, Iy**2, Ix*Iy # Calculer les termes de la matrice de Harris

        #Appliquer une ponderation avec la fenetre rectangulaire
        if methods_wds == "rectangulaire" :
            m = np.ones((size_wds,size_wds))
            Ca = ndimage.convolve(A,m)
            Cb = ndimage.convolve(B,m)
            Cc = ndimage.convolve(C,m)

        #Appliquer une ponderation avec la fenetre gaussienne
        elif methods_wds == "gaussienne" :
            Ca = ndimage.gaussian_filter(A, sigma=std_dev) # sigma : écart type
            Cb = ndimage.gaussian_filter(B, sigma=std_dev)
            Cc = ndimage.gaussian_filter(C, sigma=std_dev)
        else :
            raise TypeError(
            "`methods_wds` must be rectangulaire or gaussienne ")
         # La carte des coins de Harris
        C = (Ca*Cb - Cc**2) - K*(Ca+Cb)**2

        # Trouver les coordonnées des points d'intérêt
        kpnt = np.argwhere(C>0.1*C.max())

        # Supprimer les points d'intérêt locaux non maximaux si demendé
        if remover:
            kpnt = self.remove_local_non_maxima(gray_img, kpnt, size=suppression_size)

        return C, kpnt
    

    def haris_detector_hessienne(self, input_img, remover = False, size_suppression=3):
        """
            Détecte les points d'intérêt dans une image en utilisant le détecteur de Harris basé sur la hessienne.

            Args:
                input_img (ndarray): Image d'entrée.
                remover (bool): Si True, supprime les points d'intérêt locaux non maximaux. Par défaut False.

            Returns:
                ndarray: Carte des coins de Harris basée sur la hessienne pour l'image d'entrée.
                ndarray: Tableau NumPy contenant les coordonnées des points d'intérêt détectés.

        """

        gray_img = self.convert_to_grayscale(input_img) # Convertir l'image en niveaux de gris

        Ix, Iy = self.get_gradient(gray_img) # Calculer les gradients horizontaux et verticaux de l'image

        # Calculer les dérivées secondes par rapport à x et y 
        Ixx = np.gradient(Ix, axis = 1)
        Ixy = np.gradient(Ix, axis = 0)
        Iyy = np.gradient(Iy, axis = 0)

        # Calculer la matrice de Harris basée sur la hessienne
        H = (Ixx*Iyy - Ixy**2)
        kpnt = np.argwhere(H>0.1*H.max())

        # Trouver les coordonnées des points d'intérêt
        if remover:
            kpnt = self.remove_local_non_maxima(gray_img, kpnt, size_suppression)

        return H, kpnt
    



    def fast_detector(self, input_img, remover = False, t = 0.2 , n = 12):
        """
            Détecte les points d'intérêt dans une image en utilisant le détecteur de Fast.

            Args:
                input_img (ndarray): Image d'entrée.
                remover (bool): Si True, supprime les points d'intérêt locaux non maximaux. Par défaut False.
                t (float): Paramètre de seuillage qui sera utilisé pour comparer le pixel central par rapport à ces voisins.
                n (int): nombre de pixels successifs sur le cercle des voisins qui doivent satisfaire les conditions de seuillage? 
                
            Returns:
                ndarray: Tableau NumPy contenant les coordonnées des points d'intérêt détectés.

        """


        gray_img = self.convert_to_grayscale(input_img)  # convertir l'image en niveaux de gris

        # indices des voisins d'un pixel central sur un cercle de 16 points
        voisins = np.array([[0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1],
                            [3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3]])
        
        # générer les indices de tous les pixels de l'image
        [X,Y] = np.meshgrid(np.arange(0,gray_img.shape[0]),np.arange(0,gray_img.shape[1]))

        # offset qui permet au voisins d'un pixel (situés sur le cercle) de ne pas sortir des limites de l'images
        offset = 3

        # On ne garde que les indices qui sont décalés de l'offset pour les pixels centrales
        X = X[offset:X.shape[0]-offset,offset:X.shape[1]-offset]
        Y = Y[offset:Y.shape[0]-offset,offset:Y.shape[1]-offset]

        # On applatit les tableaux pour pouvoir les utiliser pour trouver les voisins de chaque pixel
        x = X.flatten()
        y = Y.flatten()

        # On définit xV et yV qui contiennent pour chaque pixel (x,y) les indices de ses 16 voisins situés sur un cercle dont le centre est (x,y)
        xV = x[:, np.newaxis] + voisins[0, :]
        yV = y[:, np.newaxis] + voisins[1, :]
        
        # Matrice V qui contient pour chaque pixel (x,y) les valeurs de ses 16 voisins situés sur un cercle dans le centr est (x,y)
        V = gray_img[xV,yV]

        gray_values = np.tile(gray_img[x, y], (V.shape[1], 1)).T
        # Appliquer les seuils et les conditions à la matrice V
        brighter_indices = V > gray_values + t    # si la valeur d'un voisin est supérieur à la valeur du pixel central + un seuil alors le voisin est plus lumineux
        darker_indices = V < gray_values - t      # si la valeur d'un voisin est inférieur à la valeur du pixel central + un seuil alors le voisin est plus sombre
        similar_indices = ~brighter_indices & ~darker_indices  # si la valeur d'un voisin est égal à la valeur du pixel central + un seuil alors le voisin est similaire au pixel central

        V[brighter_indices] = 1   # Les voisins qui sont plus lumineux que le pixel central + seuil prennent la valeur 1
        V[darker_indices] = -1    # Les voisins qui sont plus sombres que le pixel central + seuil prennent la valeur -1
        V[similar_indices] = 0    # Les voisins qui sont égaux aux pixel central + seuil prennent la valeur 0

        
        # mask qui contient que des 1 et dont la taille représente le nombre de voisins successives qu'on désire etre de meme type (plus lumineux ou plus sombres)
        mask = np.ones((1,n))
        mask = np.array(mask)

        V = V.reshape(V.shape[0],1,V.shape[1])  # adapte les dimensions du tableau pour pouvoir le convoluer avec le mask

        # Convolution avec le mask

        for i in range(V.shape[0]):
            V[i] = convolve2d(V[i], mask, mode='same',boundary = 'wrap')

        V = V.reshape(V.shape[0],V.shape[2])
    
        # Trouver les indices où V est égal à n ou -n ce qui correspond à 12 voisins sucessifs qui sont brighter ou darker
        indices = np.where((V == n) | (V == -n))[0]
        # S'il y'a n voisins successifs qui sont brighter ou darker dans une des ligne on la garde un seule fois
        indices = np.unique(np.array(indices))

        # Récupérer les coordonnées des pixels qui possèdent n successifs voisins plus lumineux ou plus sombres
        x = x[indices]
        y = y[indices]  
        
        
        kpnt = [x, y]
        kpnt = np.array(kpnt)
        kpnt = kpnt.transpose()

        # Si remover = True on rajoute ka suppression des non_maximas locaux
        if remover:
            kpnt = self.remove_local_non_maxima(gray_img, kpnt)

        # Return des points d'intérets
        return  kpnt
    

class Matching():
    """
        Une classe pour la correspondance de points clés en utilisant des descriptions de voisinage local.

        Args :
            - image1 : ndarray
                La première image d'entrée.
            - image2 : ndarray
                La deuxième image d'entrée.
            - keypoints_image1 : ndarray
                Points clés dans la première image.
            - keypoints_image2 : ndarray
                Points clés dans la deuxième image.
            - neighbor_size : int, optionnel
                Taille du voisinage local pour la description des points clés. Par défaut, c'est 5.

        Méthodes :
            - Description_keypoints(self) -> Tuple[ndarray, ndarray]:
                Décrit les points clés dans les deux images en fonction du voisinage local.

            - matching(self) -> ndarray:
                Fait correspondre les points clés entre les deux images.

        Exemple :
        ```
            correspondant = Matching(image1, image2, keypoints_image1, keypoints_image2, neighbor_size=5)
            correspondances = correspondant.matching()
        ```

    """
    def __init__(self, image1, image2, keypoints_image1, keypoints_image2, neighbor_size = 5, seuil=0.05, metric = "Norm_L1"):
        self.image1 = image1
        self.image2 = image2
        self.keypoints_image1 = keypoints_image1
        self.keypoints_image2 = keypoints_image2
        self.neighbor_size = neighbor_size
        self.seuil = seuil
        self.metric = metric

    def Description_keypoints(self):
        """Décrit les points clés dans les deux images en fonction du voisinage local.


        Args:
            
        
        Returns:
            - V1 : ndarray
                Descriptions de voisinage local pour les points clés dans la première image.
            - V2 : ndarray
                Descriptions de voisinage local pour les points clés dans la deuxième image.
            
        """
        
        if self.neighbor_size == 3:
            neighbor = np.array([[0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1],
                            [3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3]])
        elif self.neighbor_size == 5:
            neighbor = np.array([[0,0,1,1,1,0,-1,-1,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1],
                     [0,1,1,0,-1,-1,-1,0,1,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1,0,1,2,2]])
        elif self.neighbor_size == 7:
            neighbor = np.array([[0,0,1,1,1,0,-1,-1,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1],
                     [0,1,1,0,-1,-1,-1,0,1,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1,0,1,2,2]])
            raise TypeError(
            "`neighbor_size` must be an int ")
        
        gray_img1 = ski.color.rgb2gray(self.image1)
        gray_img2 = ski.color.rgb2gray(self.image2)

        keypoints_image1x, keypoints_image1y = self.keypoints_image1[:,0], self.keypoints_image1[:,1]
        keypoints_image2x, keypoints_image2y = self.keypoints_image2[:,0], self.keypoints_image2[:,1]
        
        xV1, yV1 = keypoints_image1x[:, np.newaxis] + neighbor[0], keypoints_image1y[:, np.newaxis] + neighbor[1]
        xV2, yV2 = keypoints_image2x[:, np.newaxis] + neighbor[0], keypoints_image2y[:, np.newaxis] + neighbor[1]


        V1 = gray_img1[xV1, yV1]
        V2 = gray_img2[xV2, yV2]
        return V1, V2
    
    def matching(self):
        """ Fait correspondre les points clés entre les deux images.


            Returns:
                - correspondances : ndarray
                    Tableau des points clés correspondants.
        """
        V1, V2 = self.Description_keypoints()
        V1_ = V1.reshape(V1.shape[0],1,V1.shape[1])
        if self.metric == "Norm_L1":
            mad = np.sum(np.abs(V1_-V2),axis = 2)
        if self.metric == "Norm_L2":
            mad = np.sqrt(np.sum((V1_-V2)**2, axis = 2))
        indices1 = np.arange(self.keypoints_image1.shape[0])
        indices2 = np.argmin(mad, axis=1)
        matches1 = np.argmin(mad, axis=0)

        # Cette partie est incpiré du code match Skimage
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]
        mask = mad[indices1, indices2] <= mad.max()*self.seuil
        indices1 = indices1[mask]
        indices2 = indices2[mask]
        matches = np.column_stack((indices1, indices2))

        return matches