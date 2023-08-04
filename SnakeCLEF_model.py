# Files/folders required to run algorithm:
# SnakeCLEF 2023 small-size training set image folder stored as SnakeCLEF_training_set/SnakeCLEF2023-small_size
# SnakeCLEF 2023 small-size validation set image folder stored as SnakeCLEF_validation_set/SnakeCLEF2023-small_size
# SnakeCLEF 2023 training set metadata stored as SnakeCLEF2023-TrainMetadata-iNat.csv
# SnakeCLEF 2023 validation set metadata stored as SnakeCLEF2023-ValMetadata.csv
# SnakeCLEF 2023 venomous status list stored as venomous_status_list.csv

# On my laptop, this algorithm takes around four and a half hours to run.
import time
start_time = time.time()

import random
import os
import itertools
import numpy as np
import pandas as pd
import sklearn.gaussian_process as skl_gp
import sklearn.gaussian_process.kernels as skl_gp_k
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

data_train = pd.read_csv('SnakeCLEF2023-TrainMetadata-iNat.csv')
code_list = np.sort(np.append(data_train['code'].unique(), np.array(['AS','CW']))).tolist() # This list contains all of the country codes in the training and validation sets.
num_codes = len(code_list)
num_species = np.max(data_train['class_id']) + 1

class SnakeCLEFFormatDataset(Dataset):
    def __init__(self, table, image_folder, transform=None):
        self.image_folder = image_folder
        self.table = table
        self.transform = transform
        self.images = []
        self.endemics = []
        self.codes = []
        self.species = []
        
        for i in self.table.index:
            folders = self.table.loc[i, 'image_path'].split(sep='/')
            file_name = folders[2]
            path_folders = self.image_folder + '\\' + folders[0] + '\\' + folders[1]
            if file_name in os.listdir(path_folders): # Check that the file is in the image folder (some instances in the table may lack images).
                try:
                    image_data = read_image(self.image_folder + '/' + self.table.loc[i, 'image_path'])
                except RuntimeError: # This is needed because some of the PNG files cannot be read by PyTorch.
                    pass
                else:
                    if len(image_data) == 3: # Some images have a number of channels different from 3, so this is necessary.
                        self.images.append(self.table.loc[i, 'image_path'])
                        self.endemics.append(self.table.loc[i, 'endemic'])
                        self.codes.append(self.table.loc[i, 'code'])
                        self.species.append(self.table.loc[i, 'class_id'])
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = read_image(self.image_folder + '/' + self.images[idx])
        image = convert_image_dtype(image) # Convert to float32.
        if self.transform:
            image = self.transform(image)
            
        image_x = len(image[0][0])
        image_y = len(image[0])
        image_size = torch.tensor([image_x, image_y]) # This is not necessary, but it is useful to have.
        
        metadata_np = np.zeros(num_codes + 1)
        if self.endemics[idx]:
            metadata_np[0] = 1 # First element of metadata tensor stores endemic status.
        if self.codes[idx] in code_list:
            code_no = code_list.index(self.codes[idx])
            metadata_np[code_no + 1] = 1 # Remaining elements of metadata tensor store country code in "yes/no" format.
        else:
            metadata_np[num_codes] = 1 # Stores any unfamiliar codes as "unknown". This would be needed if we were using a separate test set, but since we are testing on the validation set, this is not used.
        metadata = torch.tensor(metadata_np).to(torch.float32)
        
        label_scalar = self.species[idx]
        label = torch.tensor(label_scalar)
        
        return image, image_size, metadata, label

class Net(nn.Module):
    # Network architecture: convolution (7x7 kernel, 32 output feature maps) -> ReLU -> max pool (2x2 kernel, stride 2) 
    #                    -> convolution (5x5 kernel, 32 output feature maps) -> ReLU -> max pool (2x2 kernel, stride 2) 
    #                    -> convolution (3x3 kernel, 32 output feature maps) -> ReLU -> adaptive max pool (3x3 output) -> flatten, append metadata 
    #                    -> fully connected layer (1024 outputs) -> ReLU -> fully connected layer (1024 outputs) -> ReLU 
    #                    -> fully connected layer (1784 outputs, one for each species) 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.adpool = nn.AdaptiveMaxPool2d(3)
        self.fc1 = nn.Linear(32 * 3 * 3 + num_codes + 1, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_species)
    
    def forward(self, image, metadata):
        x = self.pool(F.relu(self.conv1(image)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adpool(F.relu(self.conv3(x)))
        x = torch.unsqueeze(torch.cat((x.reshape(32 * 3 * 3), torch.squeeze(metadata))), 0) # Append metadata before the first fully connected layer.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Normalize((0.5), (0.5))

print('Loading training dataset.')
snake_clef = SnakeCLEFFormatDataset(data_train, 'SnakeCLEF_training_set\\SnakeCLEF2023-small_size', transform=transform)
print('Training dataset loaded.')

data_val = pd.read_csv('SnakeCLEF2023-ValMetadata.csv')
print('Loading validation dataset.')
snake_clef_val = SnakeCLEFFormatDataset(data_val, 'SnakeCLEF_validation_set\\SnakeCLEF2023-small_size', transform=transform)
print('Validation dataset loaded.')

vsl = pd.read_csv('venomous_status_list.csv')
is_venomous = np.zeros(num_species)

for i in vsl.index: # Load data about which species are venomous.
        if vsl.loc[i, 'MIVS'] == 1:
            is_venomous[vsl.loc[i, 'class_id']] = 1

def weights(venom_weight): # Alter weight of venomous species in loss function for hyperparameter optimisation.
    output_weights_np = np.ones(num_species)
    
    for i in vsl.index:
        if is_venomous[i] == 1:
            output_weights_np[i] = venom_weight
    
    output_weights = torch.tensor(output_weights_np).to(torch.float32)
    return output_weights

def train(dataset, lr, momentum, venom_weight, data_fraction):
    data_number = len(dataset) // data_fraction # Number of images to be trained on
    completion_counter = 1 # Used for logging percentage of training done. Counts up in 10% intervals on optimisation runs and 1% intervals on final run.
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    loader_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    # Batch size is 1 because collate function does not work with input data of variable sizes.
    # Larger batches may be possible with a custom collate function but I have not been able to achieve this.
    # Multi-processing is not used because it has failed to work for me in the past.
    output_weights = weights(venom_weight)
    criterion = nn.CrossEntropyLoss(weight=output_weights)
        
    for i, data in enumerate(loader_train, 0):
        images, image_sizes, metadata, labels = data
        optimizer.zero_grad() # Train network using SGD and backprop.
        outputs = net(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if data_fraction == 20:
            if i == (completion_counter * data_number) // 10:
                print('Training ' + str(completion_counter*10) + '% complete.')
                completion_counter += 1
        elif data_fraction == 2:
            if i == (completion_counter * data_number) // 100:
                print('Training ' + str(completion_counter) + '% complete.')
                completion_counter += 1
        if i == data_number:
            break
    
    return net

index_func = lambda x, l: l[x]
v_index_func = np.vectorize(index_func, signature='(),(n)->()') # Vectorised indexing function used in testing. 

def test(dataset, alg, class_probs=True):
    print('Starting testing.')
    loader_test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0) # See above. Code below is programmed to work with larger batches, so would continue to work if a custom collate function method were used. 
    total = 0
    correct = 0
    incorrect = 0
    incorrect_nv_p_nv = 0
    incorrect_nv_p_v = 0
    incorrect_v_p_nv = 0
    incorrect_v_p_v = 0
    
    with torch.no_grad():
        for data in loader_test:
            images, image_sizes, metadata, labels = data
            outputs = alg(images, metadata)
            if class_probs:
                _, predicted = torch.max(outputs.data, 1)
            else:
                predicted = torch.tensor([outputs]) # Used for testing benchmark algorithms. 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            incorrect += (predicted != labels).sum().item()
            np_labels = labels.numpy()
            np_predicted = predicted.numpy()
            venom_labels = v_index_func(np_labels, is_venomous)
            venom_predicted = v_index_func(np_predicted, is_venomous)
            incorrect_nv_p_nv += np.count_nonzero(np.logical_and(np.logical_and(np.equal(venom_labels, 0), np.equal(venom_predicted, 0)), np.not_equal(np_predicted, np_labels)))
            incorrect_nv_p_v += np.count_nonzero(np.logical_and(np.equal(venom_labels, 0), np.equal(venom_predicted, 1)))
            incorrect_v_p_nv += np.count_nonzero(np.logical_and(np.equal(venom_labels, 1), np.equal(venom_predicted, 0)))
            incorrect_v_p_v += np.count_nonzero(np.logical_and(np.logical_and(np.equal(venom_labels, 1), np.equal(venom_predicted, 1)), np.not_equal(np_predicted, np_labels)))
            
    loss_result = incorrect_nv_p_nv + 2 * incorrect_nv_p_v + 5 * incorrect_v_p_nv + 2 * incorrect_v_p_v # Loss function for SnakeCLEF
    score = loss_result / total # Loss function divided by number of instances
    
    print('Testing complete.')
    print('Of ' + str(total) + ' instances tested, ' + str(correct) + ' were correct and ' + str(incorrect) + ' were incorrect.')
    print('Of the incorrect instances:')
    print(str(incorrect_nv_p_nv) + ' instances were of non-venomous snakes incorrectly predicted as non-venomous snakes.')
    print(str(incorrect_nv_p_v) + ' instances were of non-venomous snakes incorrectly predicted as venomous snakes.')
    print(str(incorrect_v_p_nv) + ' instances were of venomous snakes incorrectly predicted as non-venomous snakes.')
    print(str(incorrect_v_p_v) + ' instances were of venomous snakes incorrectly predicted as venomous snakes.')
    print('The percentage of instances classified correctly was ' + str(100 * correct / total) + '%.')
    print('The score for this model is ' + str(score) + ' (lower is better).')
    
    return score

def train_test(train_set, test_set, lr, momentum, venom_weight, data_fraction):
    alg = train(train_set, lr, momentum, venom_weight, data_fraction)
    score = test(test_set, alg)
    return score # The hyperparameter optimisation aims to minimise the SnakeCLEF loss function, so we return it as output from testing.

def edge_dist(l): # Edge distance function, used in the optimiser to prevent "out-of-bounds" results.
    dist = 0.5
    for i in l:
        if i < 0 or i > 1:
            return -1000000000000000000000
        elif i < dist or 1-i < dist:
            dist = min(i, 1-i)
    return dist

v_edge_dist = np.vectorize(edge_dist, signature='(n)->()')

simple_kernel = skl_gp_k.RBF(length_scale=0.1, length_scale_bounds='fixed') # The hyperparameter optimiser uses a Gaussian process surrogate model with an RBF kernel.
simple_model = skl_gp.GaussianProcessRegressor(kernel = simple_kernel, alpha=0)

def optimise_3d(data_x, data_y, size, scale, beta, param):
    simple_model.fit(data_x, data_y)
    centre = [0.5, 0.5, 0.5]
    width = 0.5
    while width > 0.0000001: # Use recursive grid search to predict best hyperparameters.
        grid = tuple(itertools.product(np.linspace(centre[0]-width, centre[0]+width, size), np.linspace(centre[1]-width, centre[1]+width, size), np.linspace(centre[2]-width, centre[2]+width, size)))
        edge_dist_grid = v_edge_dist(grid)
        mean, std = simple_model.predict(grid, return_std = True)
        ac_func = np.add(np.add(mean, np.multiply(beta, std)), np.multiply(param, edge_dist_grid))
        centre = grid[np.argmax(ac_func)]
        width *= scale
    return centre

lr = 0.05001 # Start with parameters slightly off-centre because optimiser can work awkwardly if input is on a grid point.
momentum = 0.5001
venom_weight = 2.333
x_list = []
x_list_scaled = []
y_list = []
y_list_inv = []
    
for j in range(8):
    x_list.append([lr, momentum, venom_weight])
    x_list_scaled.append([10 * lr, momentum, venom_weight / 4]) # Scale parameters to use as input for optimiser.
    print('Starting training on optimisation run ' + str(j+1) + ' of 8.')
    print('Hyperparameters: learning rate = ' + str(lr) + ', momentum = ' + str(momentum) + ', venom weight = ' + str(venom_weight))
    y = train_test(snake_clef, snake_clef_val, lr, momentum, venom_weight, 20) # Optimise hyperparameters using eight low-fidelity (0.05 epoch) runs.
    y_list.append(y)
    y_list_inv.append(2.5 - y) # Optimiser is designed for maximisation of positive functions, so we use this transformation.
        
    if j <= 6:
        print('Optimising hyperparameters.')
        edge_param = 0.000001  
        # Use smaller values of beta for later runs. 
        # For runs 2-4, use beta=50. For runs 5-6, use beta=5. For run 7, use beta=1.5. For run 8, use beta=0.
        # The model needs to start very exploratory because of the lack of data and the large amount of noise.
        if j <= 2: 
            beta = 50
        elif j <= 4:
            beta = 5
        elif j <= 5:
            beta = 1.5
        else:
            beta = 0
            
        while edge_param < 10000: # Try increasingly larger edge parameters until result is within bounds.
            new_params = optimise_3d(x_list_scaled, y_list_inv, 23, 7 / 23, beta, edge_param)
            if not (new_params[0] < 0.001 or new_params[1] < 0.001 or new_params[1] > 0.999 or new_params[2] < 0.001):
                print('Optimisation complete.')
                break
            edge_param *= 10
        if edge_param >= 10000:
            print('Optimisation failed. Using random hyperparameters.')
            new_params = [random.random(), random.random(), random.random()]
            
        lr = new_params[0] / 10
        momentum = new_params[1]
        venom_weight = new_params[2] * 4

best_params = x_list[np.argmin(y_list)] # Use best hyperparameters from low-fidelity runs for final algorithm.
print('Starting training on final model.')
print('Hyperparameters: learning rate = ' + str(best_params[0]) + ', momentum = ' + str(best_params[1]) + ', venom weight = ' + str(best_params[2]))
train_test(snake_clef, snake_clef_val, best_params[0], best_params[1], best_params[2], 2) 
# Train final algorithm for 0.5 epochs.
# The public test set does not have the species labels available, so we test on the validation set. Therefore the final result is slightly overfitted to the validation set.
# A typical final result is 6.5% of snakes predicted correctly with a score of 1.65. Because of the randomness in the training, the result can sometimes be considerably worse.

print('Benchmark 1: predict all instances randomly.')
random_alg = lambda images, metadata: random.randint(0, 1783)
test(snake_clef_val, random_alg, class_probs=False)

print('Benchmark 2: predict Natrix natrix (most common species in dataset) for all instances.')
natrix = lambda images, metadata: 1137
test(snake_clef_val, natrix, class_probs=False)

print('Benchmark 3: predict Vipera berus (most common venomous species in dataset) for all instances.')
vipera = lambda images, metadata: 1736
test(snake_clef_val, vipera, class_probs=False)

time_taken = time.time() - start_time
time_taken_hours = time_taken // 3600
time_taken_minutes = (time_taken - 3600 * time_taken_hours) // 60
print('Algorithm finished running after ' + str(int(time_taken_hours)) + ' hours and ' + str(int(time_taken_minutes)) + ' minutes.')
