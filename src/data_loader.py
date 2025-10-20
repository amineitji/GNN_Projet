import torch
import networkx as nx
from torch_geometric.utils import from_networkx
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AirportDataLoader:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_data(self):
        G = nx.read_graphml(self.xml_path)
        data = from_networkx(G)
        
        # Extract features
        features, pops, countries, cities = [], [], [], []
        for node_id in sorted(G.nodes(), key=lambda x: int(x)):
            node = G.nodes[node_id]
            lat = float(node.get('lat', 0))
            lon = float(node.get('lon', 0))
            pop = float(node.get('population', 0))
            degree = G.degree(node_id)
            
            features.append([lat, lon, np.log1p(pop), degree])
            pops.append(pop)
            countries.append(node.get('country', 'UNKNOWN'))
            cities.append(node.get('city_name', 'UNKNOWN'))
        
        # Normalize
        features = np.array(features)
        features[:, :2] = self.scaler.fit_transform(features[:, :2])
        
        # Create tensors
        data.x = torch.FloatTensor(features)
        data.population = torch.FloatTensor(pops)
        data.country_labels = self.label_encoder.fit_transform(countries)
        data.country_names = countries
        data.city_names = cities
        
        # Splits
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