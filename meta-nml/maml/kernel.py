import numpy as np
from scipy.spatial import distance


class KernelEmbedding(object):
    def __init__(self,
                 dist_weight_thresh,
                 model,
                 distance_metric = 'L2',
                ):
        """
        Class that weights a batch of points using various schemes.

        Parameters
        ----------

        dist_weight_thresh : float
            If provided, will weight points if using the L2 distance mode
        
        model
            The torch model
        
        distance_metric : str in {'cosine', 'L2'}
            The kind of distance metric to use in comparing points
        """
        self.dist_weight_thresh = dist_weight_thresh
        self.model = model
        self.embedding = self.model.embedding if hasattr(self.model, 'embedding') else lambda x: x
        if distance_metric == 'L2':
            self.distance_metric = self.l2_dist
        elif distance_metric == 'cosine':
            self.distance_metric = self.cosine_dist
        else:
            raise NotImplementedError
    
    def cosine_dist(self, x, y):
        x = [ x.numpy() ]
        # print(x.shape, y.shape)
        return distance.cdist(x, y, metric = 'cosine')[0]

    def l2_dist(self, x, y):
        return np.linalg.norm(x - y, axis = -1)
    
    def embed(self, query_point, batch):
        distances = self.distance_metric( self.embedding(query_point), self.embedding(batch))
        return self.weights_from_distances(distances)
    
    def weights_from_distances(self, distances):
        """
        Returns weighting based on negative exponential distance.
        """
        return np.exp(-distances * 2.3 / self.dist_weight_thresh)
    
    def dirac_loss(self, query_point, batch):
        """
        An extreme weighting scheme where only copies of the query point would get weighted, and everything else would be weighted 0.
        """
        raise NotImplementedError # would likely return [1, 0, 0, 0, ... ] with 1s for any other copies of the query
    

    # def weight_queries(self, query_point, X, mode = "cosine"):
    #     if mode == "L2":
    #         # self.weight_embedding = lambda x: x
    #         return np.exp(-np.linalg.norm(self.weight_embedding(query_point) - self.weight_embedding(X), axis=-1) * 2.3 / self.dist_weight_thresh)
    #     elif mode == 'cosine':
    #         print(query_point, X)
    #         # query_point =  #query_point.cuda()
    #         # model = self.model #.cpu()
    #         embedded_query = self.weight_embedding(query_point.cuda()).cpu().numpy() #self.weight_embedding(query_point)
    #         embedded_examples = self.weight_embedding(X.cuda()).detach().cpu()
    #         vectorized_query = np.tile(embedded_query, (X.shape[0], 1 ))
    #         distances = np.diagonal(distance.cdist(vectorized_query, embedded_examples, 'cosine'))
    #         return np.exp(-distances * 2.3 / self.dist_weight_thresh)     
    #     else:
    #         raise NotImplementedError
    
    # def set_model(self, model):
    #     self.model = model
