from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    Note here each iteration indicates a batch of samples. 
    Thus, this BatchSampler must used together with a batch sampler.
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter 
    
    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # after iterating over all the samples in the dataset
            # the second time we call the loop, the iter method in
            # batch sampler, e.g., RandomSampler, will reshuffle the
            # indices of the dataset and create a new iterable object.
            for batch in self.batch_sampler:
                iteration += 1

                if iteration > self.num_iterations:
                    break
                yield batch 
    
    def __len__(self):
        return self.num_iterations
                
