import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Subset

def loaders_by_classes(dataset, batch_size, shuffle=True, num_workers=0):
    """
    input : dataset (torch.utils.data.Dataset) -  dataset to split by classes

    output : data_loaders (dict - class : torch.utils.data.DataLoader) -  keys are class names, values are DataLoaders for each class
    """
    # Reverse the class_to_idx dictionary to get idx_to_class
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Dict of lists to hold indices for each class
    class_indices = {_ : [] for _ in idx_to_class.keys()}

    # Iterate once through the dataset to collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Create DataLoaders for each class
    data_loaders = {idx_to_class[class_idx] : DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True, num_workers=0)
                    for class_idx, indices in class_indices.items()}
    
    return data_loaders

def balance(loaders):
    """
    input : loaders (dict - class : torch.utils.data.DataLoader) - dictionnaire avec les classes en clés et DataLoader comme valeurs.

    output : class_counts (dict - class : int) - dictionnaire avec les classes en clés et la quantité de données pour chaque classe en valeurs.
    """
    class_counts = {}

    for class_name, loader in loaders.items():
        # Accéder directement à la taille du dataset sous-jacent via l'attribut `dataset`
        class_counts[class_name] = len(loader.dataset)

    return class_counts

def balanced_batch_size(loaders, class_name):
    bal = balance(loaders)
    n_items_class = bal[class_name]
    n_tot = 0
    for key, val in bal.items():
        if key != class_name:
            n_tot += val
    proportion = n_items_class / n_tot
    balcd_batch_size = (1 - proportion) / proportion
    return balcd_batch_size

def filter_loaders(loaders, excluded_classes, batch_size, shuffle=True, num_workers=0):
    """
    input : 
        loaders (dict) - Dictionnaire avec les classes comme clés et DataLoader comme valeurs.
        excluded_classes (list) - Liste des classes à exclure du DataLoader.
    
    output : train_loader (torch.utils.data.DataLoader) - Nouveau DataLoader combiné avec les classes filtrées.
    """
    # On récupère tous les indices des loaders pour les classes qui ne sont pas exclues
    filtered_indices = []
    dataset = None  # On utilisera cette variable pour retrouver le dataset d'origine

    for class_name, loader in loaders.items():
        if excluded_classes is not None :
            if class_name not in excluded_classes:
                # Concatène tous les indices du loader pour les classes non exclues
                filtered_indices += loader.dataset.indices  # indices provenant de Subset
                dataset = loader.dataset.dataset  # Référence au dataset d'origine
        else :
            filtered_indices += loader.dataset.indices
            dataset = loader.dataset.dataset
    
    # Créer un nouveau DataLoader avec les indices filtrés
    if dataset is not None:
        combined_loader = DataLoader(Subset(dataset, filtered_indices), batch_size=loader.batch_size, shuffle=shuffle, num_workers=num_workers)
        return combined_loader
    else:
        return None
    
def get_logits(class_name, model, loaders, device):
    loader = loaders[class_name]
    model.eval()

    # Sélectionner un batch aléatoire
    loader_list = list(loader)  # Convertir le DataLoader en liste pour pouvoir indexer
    random_batch = random.choice(loader_list)  # Choisir un batch aléatoire

    data, _ = random_batch  # Extraire les données (ignorer les labels)
    data = data.to(device)  # Envoyer les données sur le device

    # Passer les données à travers le modèle pour obtenir les logits
    with torch.no_grad():
        output = model(data)  # Calculer les logits pour ce batch

    return output  # Retourne les logits pour le batch sélectionné


def submodel(model, class_num):
    """
    Crée un sous-modèle basé sur le modèle donné en omettant une classe spécifique,
    sans dépendre des noms explicites des couches.
    
    Args:
        model: Modèle PyTorch à modifier.
        class_num: Numéro de la classe à omettre.
        
    Returns:
        Nouveau modèle avec une sortie réduite.
    """
    # Identifier automatiquement le nombre de classes en inspectant la dernière couche
    last_layer_name = None
    last_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # Cherche la dernière couche linéaire
            last_layer_name = name
            last_layer = module

    if last_layer is None:
        raise ValueError("Aucune couche linéaire trouvée dans le modèle.")
    
    original_num_classes = last_layer.out_features
    if class_num < 0 or class_num >= original_num_classes:
        raise ValueError(f"Le numéro de classe doit être compris entre 0 et {original_num_classes - 1}.")
    
    # Créer une copie du modèle
    new_model = type(model)()  # Initialise un nouveau modèle de même type
    new_model.load_state_dict(model.state_dict())  # Copie les poids existants

    # Modifier la dernière couche linéaire pour avoir une sortie de taille réduite
    new_last_layer = nn.Linear(last_layer.in_features, original_num_classes - 1)
    
    # Copier les poids et biais de l'ancienne couche, en omettant `class_num`
    old_weights = last_layer.weight.data.clone()
    old_bias = last_layer.bias.data.clone()
    new_last_layer.weight.data = torch.cat((old_weights[:class_num], old_weights[class_num + 1:]), dim=0)
    new_last_layer.bias.data = torch.cat((old_bias[:class_num], old_bias[class_num + 1:]), dim=0)

    # Remplacer dynamiquement la dernière couche dans le modèle
    parent_module = new_model
    *parent_path, layer_to_replace = last_layer_name.split(".")
    for part in parent_path:
        parent_module = getattr(parent_module, part)
    setattr(parent_module, layer_to_replace, new_last_layer)

    return new_model

def get_activations(model, input_tensor, layer_num, verbose=True):
    """
    Get the internal representation of input tensor at after the layer 'layer_num'
    """
    
    layers = list(model.children())
    layer = layers[layer_num]

    if verbose:
        print(f'Checking residues of the input after layer : {layer}')

    # Variable pour stocker les activations
    activations = None

    # Hook pour capturer les activations
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output

    handle = layer.register_forward_hook(hook_fn)
    output = model(input_tensor)
    handle.remove()

    return {'output' : output, 'activations' : activations}