import torch
import networkx as nx
from torch_geometric.utils import from_networkx
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AirportDataLoader:
    def __init__(self, xml_path):
        """
        Initialise le chargeur de données.
        - label_encoder: Convertira les noms de pays (textuels) en indices numériques (ex: "USA" -> 0, "FRANCE" -> 1).
        - scaler: Normalisera les caractéristiques pour que le modèle s'entraîne mieux.
        """
        self.xml_path = xml_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Fonction principale pour charger et transformer les données du fichier GraphML.
        """
        # Charge le graphe depuis le fichier XML en utilisant networkx
        G = nx.read_graphml(self.xml_path)
        # Convertit le graphe networkx en un objet Data de PyTorch Geometric,
        # qui contient les arêtes (edge_index)
        data = from_networkx(G)
        
        # --- Extraction des Caractéristiques (Features) et Étiquettes (Labels) ---
        features, pops, countries, cities = [], [], [], []
        
        # Boucle sur chaque nœud (aéroport) dans le graphe
        # trié par ID pour assurer un ordre cohérent
        for node_id in sorted(G.nodes(), key=lambda x: int(x)):
            node = G.nodes[node_id]
            
            # Récupère les attributs de base
            lat = float(node.get('lat', 0))
            lon = float(node.get('lon', 0))
            pop = float(node.get('population', 0))
            degree = G.degree(node_id) # Le nombre de connexions (vols)
            
            # --- Ingénierie des caractéristiques (Feature Engineering) ---
            # Crée le vecteur de caractéristiques [x] pour chaque nœud.
            # C'est ce que le modèle utilisera pour "comprendre" le nœud.
            # On utilise np.log1p(pop) pour normaliser la population,
            # qui a une distribution très étendue (de 10k à 22M).
            features.append([lat, lon, np.log1p(pop), degree])
            
            # Stocke les étiquettes (ground truth) que le modèle essaiera de prédire
            pops.append(pop) # La vraie population (pour la régression)
            countries.append(node.get('country', 'UNKNOWN')) # Le vrai pays (pour la classification)
            cities.append(node.get('city_name', 'UNKNOWN'))
        
        # --- Normalisation ---
        features = np.array(features)
        # Normalise uniquement la latitude et la longitude (les 2 premières colonnes)
        # pour les centrer autour de 0 avec un écart-type de 1.
        features[:, :2] = self.scaler.fit_transform(features[:, :2])
        
        # --- Création des Tenseurs PyTorch ---
        # Convertit les listes numpy en Tenseurs pour PyTorch
        # data.x : Les caractéristiques d'entrée du modèle
        data.x = torch.FloatTensor(features) 
        
        # data.population : L'étiquette de régression (la vraie population)
        data.population = torch.FloatTensor(pops)
        
        # data.country_labels : L'étiquette de classification (le vrai pays, encodé en chiffres)
        data.country_labels = self.label_encoder.fit_transform(countries)
        
        # Stocke les noms pour l'évaluation finale (pour savoir quel aéroport est anormal)
        data.country_names = countries
        data.city_names = cities
        
        # --- Création des Masques Train/Validation/Test ---
        # Sépare les données pour l'entraînement (60%), la validation (20%) et le test (20%)
        # C'est crucial pour évaluer si le modèle généralise bien.
        n = data.num_nodes
        idx = np.random.permutation(n)
        train_size, val_size = int(0.6*n), int(0.2*n)
        
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask = torch.zeros(n, dtype=torch.bool)
        data.test_mask = torch.zeros(n, dtype=torch.bool)
        
        data.train_mask[idx[:train_size]] = True
        data.val_mask[idx[train_size:train_size+val_size]] = True
        data.test_mask[idx[train_size+val_size:]] = True
        
        print(f"✅ Loaded: {n} nodes, {data.num_edges} edges, {len(np.unique(data.country_labels))} countries")
        return data