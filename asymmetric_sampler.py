import math
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

    def __iter__(self):
        # Get the indices for the current rank
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        
        # Oversample or undersample based on the batch size factor
        indices = indices[:self.num_samples]
        
        return iter(indices)

    def __len__(self):
        return self.num_samples
