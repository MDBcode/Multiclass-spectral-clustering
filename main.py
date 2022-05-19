import math
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_blobs, make_circles
from PIL import Image


def msc(dataset, N, K, mi, treshold, sigma, method):
    if method == 1:
        W = gauss_function(dataset,sigma)
    elif method == 2:
        W = radius_distance(dataset,sigma)
    elif method == 3:
        W = gauss_function_img(dataset,sigma)
    print("W:\n", W)

    # korak 1: računamo degree matricu D
    e_N = [1] * N
    D = np.diag(np.matmul(W, e_N)) 
    print("D:\n", D)

    P = np.matmul(np.linalg.inv(D), W)
    print("P:\n", P)
    S, V = np.linalg.eigh(P) #računamo sv. vrijednosti i pripadne sv. vektore matrice P=D^-1 W
    print("V:\n", V, "\nS:", S)

    # korak 2: naci Z ... "rješenje u terminima sv. vektora", Z generira familiju globalnih optimuma.
    Z = np.zeros((N, K))
    for i in range(0, K):
        x = S.max() # najveća sv. vrijednost
        print("x:", x)
        max_index = np.argmax(S) # pripadni index, tamo se nalazi najveća sv. vrijednost
        print("i:", i, ", max_indeks:", max_index)
        Z[:, i] = V[:, max_index] # na i-ti stupac u Z stavi sv. vektor od pripadne nađene max sv. vrijednosti
        S[max_index] = -x #sad više x nije max(S) nego min(S) pa ga ne možemo opet uzet za max u sljedećem traženju, dakle tražimo pripadne sv. vektore od najvećeg prema najmanjem

    # korak 3: normaliziranje Z; dobivamo kao rezultat X = g(Z) = Diag(diag(ZZ^T))^-1/2 * Z, X predstavlja rješenje u neprekidnoj domeni
    print("\nZ:\n", Z, "\n")
    X = np.zeros((N, K))
    for i in range(0, N):
        dot_prod = np.dot(Z[i, :], Z[i, :]) #skalarni produkt i-tog retka iz Z sa samim sobom, efektivno kvadrira svaki element u retku
        X[i, :] = ((1 / math.sqrt(dot_prod)) * Z[i, :]) #normiraj pojedini element iz i-tog retka od Z 

    # korak 4: inicijaliziraj X; To je sad inicijalno diskretno rješenje
    print("\nX:\n", X)
    R = np.zeros((K, K)) # Ortogonalna matrica rotacije R
    rand = random.randrange(0, N-1) 
    R[:, 0] = X[rand, :] # prvi stupac u R je neki random redak iz X, ostali stupci u R su sve nule
    #na predavanjima je predloženo početi sa Q = R = I_k
    print("\nR:", R)

    # korak 5: inicijaliziramo fi za pracenje konvergencije
    c = [0] * N
    fi = 0

    # korak 6: trazimo optimalno diskretno rjesenje
    for _ in range(2, K+1):
        print(len(c))
        print(X.shape)
        print(R.shape)
        c = c + abs(np.matmul(X, R[:, K-1]))
        print(c)
        min_indeks = np.argmin(c)
        print(min_indeks)
        R[:, K-1] = X[min_indeks, :]

    # korak 7 i 8: trazimo optimalnu ortonormiranu matricu R
    rez = np.zeros(mi)
    print(rez)
    counter = -1
    f = 0
    while(counter < mi-1):
        counter = counter+1
        print("counter: ", counter)
        # neprekidno rjesenje
        X_C = np.matmul(X, R)
        # inicijaliziraj diskretno rjesenje
        X_D = np.zeros((N, K))
        # trazi diskretno rjesenje X, korak 7:
        for i in range(0, N):
            x = max(X_C[i, :]) #nađi u svakom retku neprekidnog rješenja najveći element
            max_indeks = np.argmax(X_C[i, :]) #nađi indeks u retku di je taj element
            X_D[i, max_indeks] = 1 #postavi odgovarajući elem. u diskretnom rješenju na 1, ostali u tom redu su 0

        # trazi ortonormiranu matricu R, korak 8:
        # - prvo nadi SVD faktorizaciju
        U, D, V = np.linalg.svd(np.matmul(np.transpose(X_D), X))
        D = np.diag(D)
        # - f_n je trag matrice D
        f_n = np.sum(np.diag(D))

        # - provjeri preciznost rjesenja
        rez[counter] = abs(f - f_n)
        print("rez[counter]", rez[counter])
        if rez[counter] < treshold:
            break
        # - pronadi ortogonalnu optimalnu matircu R
        f = f_n
        R = np.matmul(V, np.transpose(U))

    if rez[counter] > treshold:
        return None
    
    return X_D

def gauss_function(dataset,sigma):
    #sigma = np.std(dataset)
    print("sigma:",sigma)
    # this is an NxD matrix, where N is number of items and D its dimensionalites
    pairwise_dists = squareform(pdist(dataset, 'euclidean'))
    print("Matrica udaljenosti\n:", pairwise_dists)
    W = np.exp(-(pairwise_dists ** 2) / (2*sigma ** 2))
    return W

def gauss_function_img(dataset,sigma):
    dimx, dimy = dataset.shape
    arr = dataset.flatten()
    M = M2 = np.zeros((dimx*dimy,2))
    M[:,0] = arr
    print("M:",M)
    pairwise_color_diff = squareform(pdist(M, 'euclidean')) # pdist računa vektor udaljenosti izmedu točaka(npr 9x1), squareform prebaci vektor u 3x3 matricu

    W = np.exp(-(pairwise_color_diff ** 2) / (2*sigma ** 2))
    #W = np.matmul(np.exp(-(pairwise_color_diff ** 2) / (2*sigma ** 2)), np.exp(-(pairwise_dists ** 2) / (2*sigma2 ** 2)))
    return W

def radius_distance(dataset,sigma):
    pairwise_dists = squareform(pdist(radius_diff(dataset), 'euclidean'))
    W = np.exp(-(pairwise_dists ** 2) / (2*sigma ** 2))
    print("W:",W)
    return W

def radius_diff(dataset):
    radii = np.array([])
    M = np.zeros((N,2))
    for (x,y) in dataset:
        print("(x,y):",(x,y))
        r = math.sqrt(x**2+y**2)
        print("r:",r)
        radii = np.append(radii,[r])
    print("radii:",radii)
    for i,r in enumerate(radii):
        M[i,0] = r
    print("M:",M)
    print("razlike radiusa:",pdist(M, 'euclidean'))
    return M


def color_graph(dataset, X, K, colormap):
    global categories

    for index, point in enumerate(dataset):
        for column in range(K):  # K particija
            if X[index][column] != int(0):
                # boja, zapravo odredi kategoriju (cluster) kojem pripada tocka
                categories[index] = column
    plt.scatter(dataset[:, 0], dataset[:, 1], s=100, c=colormap[categories])
    plt.show()

def color_image(dataset, X, K, colormap):
    global categories
    dimx,dimy = dataset.shape

    X_1 = X[:,0]
    X_2 = X[:,1]
    X_1 = np.reshape(X_1,(dimx,dimy))
    X_2 = np.reshape(X_2,(dimx,dimy))

    for i in range(dimx):
        for j in range(dimy):
            if X_1[i][j] == 1:
                dataset[i][j] = 255
            else:
                dataset[i][j] = 0
            
    im2 = Image.fromarray(dataset)
    im2 = im2.resize((256,256))
    im2.show()

def make_random_points(N):
    dataset, y = make_blobs(n_samples=N) #možemo još kao argument dodati npr centers=6, inače uvijek generira 3
    plt.scatter(dataset[:, 0], dataset[:, 1], alpha=0.7, edgecolors='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return dataset

def make_concentric_points(N):
    dataset, y = make_circles(n_samples=N, noise=0.05)
    print(dataset)
    plt.scatter(dataset[:, 0], dataset[:, 1], alpha=0.7, edgecolors='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return dataset
    

if __name__ == "__main__":
    method = int(input("Odaberi metodu 1-2-3:"))
    N = 100
    mi = 1000
    treshold = 0.001
    # inicijalno sve tocke su u prvom(nultom) clusteru
    colormap = np.array(['r', 'g', 'b', 'y', 'm', 'k', 'c'])
    if method == 1:# klasteriranje random 2D tocaka
        categories = np.array([0]*N)
        dataset = make_random_points(N)
        while True:
            K = int(input("Unesi broj clustera: "))
            sigma = float(input("Unesi sigmu: "))

            X = msc(dataset, N, K, mi, treshold, sigma, method)
            if X is None:
                print("Metoda nije uspjela!")
            else:
                print("Matrica particije:\n", X)
                color_graph(dataset, X, K, colormap)

    elif method == 2:# klasteriranje 2D koncentricnih tocaka
        categories = np.array([0]*N)
        dataset = make_concentric_points(N)
        while True:
            K = int(input("Unesi broj clustera: "))
            sigma = float(input("Unesi sigmu: "))
            X = msc(dataset, N, K, mi, treshold, sigma, method)
            if X is None:
                print("Metoda nije uspjela!")
            else:
                print("Matrica particije:\n", X)
                color_graph(dataset, X, K, colormap)

    elif method == 3:# segmentacija slike    
        path = "cat.jpg"
        dim = 64
        image = Image.open(path).resize((dim,dim)).convert('L')
        img = np.array(image)
        dataset = img/255

        dimx,dimy = dataset.shape
        categories = np.zeros((dimx,dimy))
        while True:
            K = int(input("Unesi broj clustera: "))
            sigma = float(input("Unesi sigmu: "))

            X = msc(dataset, dimx*dimy, K, mi, treshold, sigma, method)
            if X is None:
                print("Metoda nije uspjela!")
            else:
                print("Matrica particije:\n", X)
                color_image(dataset, X, K, colormap)