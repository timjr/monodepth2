import math
import torch
from torch.utils.data.distributed import DistributedSampler

class AsymmetricDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, batch_size_factors=None, **kwargs):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, **kwargs)
        self.batch_size_factors = batch_size_factors or [1] * self.num_replicas
        self.total_factor = sum(self.batch_size_factors)
        self.num_samples_per_replica = [
            int(math.ceil(len(self.dataset) * factor / self.total_factor))
            for factor in self.batch_size_factors
        ]
        self.total_samples = sum(self.num_samples_per_replica)
        
        # Adjust the total number of samples and indices based on factors
        self.num_samples = self.num_samples_per_replica[self.rank]
        print("in the asym constructor, the batch size is", self.num_samples)
        

    def __iter__(self):
        # Generate a list of indices for the dataset
        indices = list(range(len(self.dataset)))
        
        # Shuffle indices
        if self.shuffle:
            # Ensure all replicas use the same random seed for shuffling
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(len(indices), generator=g).tolist()
        
        # Calculate the starting index for this rank
        start_idx = sum(self.num_samples_per_replica[:self.rank])
        end_idx = start_idx + self.num_samples
        
        # Select the subset of indices for this rank
        indices = indices[start_idx:end_idx]
        
        # Optionally oversample/undersample if needed to match the shard size
        if len(indices) < self.num_samples:
            diff = self.num_samples - len(indices)
            print(f"oversampling by {diff} to meet expected shard size")
            indices += indices[:diff]
        else:
            indices = indices[:self.num_samples]
        
        return iter(indices)

    def __len__(self):
        return self.num_samples
