import torch
import pdb

class StaticFeatureAttributionPrior:
    def __init__(self, explainer, prior_feature, ignored_features, background_dataset):
        super(StaticFeatureAttributionPrior, self).__init__()
        self.explainer = explainer
        self.prior_feature = prior_feature
        self.ignored_features = ignored_features
        self.background_dataset = background_dataset

    def penalty(self, shap_values):
        #pdb.set_trace()
        prior_feature = torch.FloatTensor(self.prior_feature).cuda()
        eg_no_drugs = shap_values[:, len(self.ignored_features):]
        prior_penalty = torch.sum(((eg_no_drugs * 1000 - prior_feature) ** 2), dim=1) / 100000

        return prior_penalty
