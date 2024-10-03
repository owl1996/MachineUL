# Function to split dataset by all classes in one pass
def split_by_all_classes(dataset, num_classes=10):
    class_indices = [[] for _ in range(num_classes)]  # List of lists to hold indices for each class

    # Iterate once through the dataset to collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Create DataLoaders for each class
    data_loaders = [DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True, num_workers=0)
                    for indices in class_indices]
    
    return data_loaders