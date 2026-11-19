# Importazione delle librerie
import numpy as np
import matplotlib.pyplot as plt

# Importazione degli strumenti di Scikit-Learn
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. CARICAMENTO DEL DATASET
# ==================================================

# Carichiamo il dataset Iris
data = load_iris()

# X contiene le caratteristiche dei fiori
# y contiene le etichette delle specie originali
# target_names contiene i nomi delle tre specie
X = data.data
y = data.target
target_names = data.target_names

# Stampiamo le dimensioni dei dati
print(X.shape)
print(y.shape)
print(target_names)


# ==================================================
# 2. STANDARDIZZAZIONE
# ==================================================

# Creiamo l'oggetto StandardScaler
scaler = StandardScaler()

# Standardizziamo le quattro caratteristiche
# Ogni caratteristica avrà media 0 e deviazione standard 1
X_std = scaler.fit_transform(X)


# ==================================================
# 3. RIDUZIONE DIMENSIONALE CON PCA
# ==================================================

# Creiamo una PCA che riduce le caratteristiche a 2 componenti
pca = PCA(n_components=2)

# Trasformiamo i dati standardizzati
# Passiamo da 4 caratteristiche a 2 componenti principali
X_pca = pca.fit_transform(X_std)

# Stampiamo le dimensioni prima e dopo la PCA
print("Forma dati standardizzati:", X_std.shape)
print("Forma dati dopo PCA:", X_pca.shape)


# ==================================================
# 4. RICERCA DEL NUMERO DI COMPONENTI GMM
# ==================================================

# Lista vuota che conterrà i valori di log-likelihood
log_likelihood = []

# Proviamo un numero di componenti da 1 a 9
n_components = range(1, 10)

for k in n_components:

    # Creiamo un modello Gaussian Mixture
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42
    )

    # Addestriamo il modello sui dati standardizzati
    gmm.fit(X_std)

    # Calcoliamo e salviamo la log-likelihood media
    # score() restituisce la log-likelihood media per campione
    log_likelihood.append(gmm.score(X_std))


# Stampiamo i valori ottenuti
print(log_likelihood)


# ==================================================
# 5. VISUALIZZAZIONE ELBOW METHOD
# ==================================================

plt.figure(figsize=(8, 6))

# Creiamo il grafico della log-likelihood
plt.plot(
    n_components,
    log_likelihood,
    marker="o"
)

plt.title("Elbow Method log-likelihood")
plt.xlabel("Numero di componenti (K)")
plt.ylabel("Log-likelihood media per campione")
plt.grid()

# Salviamo il grafico
plt.savefig("elbow_method_log_likelihood.png")

plt.show()


# ==================================================
# 6. ADDESTRAMENTO DEL GMM DEFINITIVO
# ==================================================

# Scegliamo il numero di componenti
best_k = 3

# Creiamo il modello definitivo
gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="full",
    random_state=42
)

# Addestriamo GMM utilizzando tutte le 4 caratteristiche
gmm.fit(X_std)

# Assegniamo ogni fiore al rispettivo cluster
labels = gmm.predict(X_std)

# Stampiamo le dimensioni e le etichette ottenute
print("Forma labels:", labels.shape)
print("Labels:", labels)


# ==================================================
# 7. CONTEGGIO DEI CLUSTER
# ==================================================

# Individuiamo gli identificativi dei cluster
# e contiamo quanti elementi appartengono a ciascuno
cluster_ids, cluster_counts = np.unique(
    labels,
    return_counts=True
)

# Stampiamo il numero di fiori per ogni componente
for cluster_id, count in zip(cluster_ids, cluster_counts):
    print(f"Componente {cluster_id}: {count} fiori")


# ==================================================
# 8. VISUALIZZAZIONE DEI CLUSTER CON PCA
# ==================================================

plt.figure(figsize=(8, 6))

# Ogni punto rappresenta un fiore
# X_pca[:, 0] rappresenta la prima componente principale
# X_pca[:, 1] rappresenta la seconda componente principale
# c=labels assegna un colore in base al cluster
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=50,
    alpha=0.7
)

plt.title("GMM Clustering (PCA)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid()

# Salviamo il grafico
plt.savefig("gmm_clustering_pca.png")

plt.show()


# ==================================================
# 9. VISUALIZZAZIONE DELLE SPECIE REALI
# ==================================================

plt.figure(figsize=(8, 6))

# Individuiamo tutte le classi reali presenti in y
for label in np.unique(y):

    # Creiamo una maschera per selezionare
    # soltanto i fiori appartenenti alla classe corrente
    mask = (y == label)

    # Rappresentiamo i fiori della classe selezionata
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=50,
        alpha=0.7,
        label=target_names[label]
    )

plt.title("Iris Dataset (PCA)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")

# Mostriamo la legenda con i nomi delle specie
plt.legend(title="Species")

plt.grid()

# Salviamo il grafico
plt.savefig("iris_dataset_pca.png")

plt.show()


# ==================================================
# 10. FUNZIONE PER CALCOLARE L'ACCURACY DEL GMM
# ==================================================

def gmm_accuracy(y_true, y_pred):

    # Creiamo un array iniziale con le etichette a zero
    mapped_labels = np.zeros_like(y_pred)

    # Analizziamo ogni cluster individuato da GMM
    for cluster_id in np.unique(y_pred):

        # Selezioniamo gli elementi appartenenti
        # al cluster corrente
        mask = (y_pred == cluster_id)

        # Recuperiamo le vere etichette degli elementi
        # appartenenti al cluster selezionato
        true_labels = y_true[mask]

        if true_labels.size > 0:

            # Contiamo quante volte compare ogni classe
            # e individuiamo quella più frequente
            most_common = np.bincount(true_labels).argmax()

            # Assegniamo a tutto il cluster
            # la classe reale più frequente
            mapped_labels[mask] = most_common

    # Confrontiamo le etichette associate con quelle reali
    # e calcoliamo la percentuale di corrispondenze
    return np.mean(mapped_labels == y_true)


# ==================================================
# 11. CALCOLO FINALE DELL'ACCURACY
# ==================================================

# Confrontiamo le specie reali con quelle associate ai cluster
accuracy = gmm_accuracy(y, labels)

# Stampiamo il risultato in percentuale con due decimali
print(f"GMM Clustering Accuracy: {accuracy * 100:.2f}%")